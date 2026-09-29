import sys
import logging
from typing import TYPE_CHECKING
from pathlib import Path
from enum import Enum
from collections import defaultdict
import time
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torchapp import TorchApp, Param, method, main, tool

from .output import print_polars_df
from .logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from torchmetrics import Metric
    from hierarchicalsoftmax import SoftmaxNode
    from torch import nn
    import lightning as L
    # import pandas as pd
    import polars as pl


def resolve_output_path(path: Path | str | bool, output_dir: Path, default_name: str) -> Path:
    """
    Resolves the path of an optional output file and creates its parent directory.

    - True, 'true' or '1' gives `output_dir / default_name`.
    - An existing directory gives `path / default_name`.
    - A bare filename (e.g. 'predictions.tsv') is placed in `output_dir`.
    - Any other path is used as given.
    """
    if isinstance(path, bool) or str(path).lower() in ("true", "1"):
        resolved = Path(output_dir) / default_name
    else:
        resolved = Path(path)
        if resolved.is_dir():
            resolved = resolved / default_name
        elif not resolved.is_absolute() and len(resolved.parts) == 1:
            resolved = Path(output_dir) / resolved

    resolved.parent.mkdir(exist_ok=True, parents=True)
    return resolved


class ImageFormat(str, Enum):
    """The image format to use for the output images."""

    NONE = ""
    PNG = "png"
    JPG = "jpg"
    SVG = "svg"
    PDF = "pdf"
    DOT = "dot"

    def __str__(self):
        return self.value

    def __bool__(self) -> bool:
        """Returns True if the image format is not empty."""
        return self.value != ""


class Barbet(TorchApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, "main_app") and hasattr(self.main_app, "info"):
            self.main_app.info.context_settings = {"help_option_names": ["-h", "--help"]}
        if hasattr(self, "tools_app") and hasattr(self.tools_app, "info"):
            self.tools_app.info.context_settings = {"help_option_names": ["-h", "--help"]}

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger") or self._logger is None:
            output_dir = getattr(self, "output_dir", Path("output"))
            self._logger = setup_logger(output_dir)
        return self._logger

    @logger.setter
    def logger(self, logger: logging.Logger) -> None:
        self._logger = logger

    @method
    def setup(
        self,
        memmap: str = None,
        memmap_index: str = None,
        treedict: str = None,
        stack_size: int = 32,        
        in_memory: bool = False,
        tip_alpha: float = None,
    ) -> None:
        if not treedict:
            raise ValueError("treedict is required")
        if not memmap:
            raise ValueError("memmap is required")
        if not memmap_index:
            raise ValueError("memmap_index is required")

        from hierarchicalsoftmax import TreeDict
        import numpy as np
        from barbet.data import read_memmap

        self.stack_size = stack_size

        self.logger.info(f"Loading treedict {treedict}")
        individual_treedict = TreeDict.load(treedict)
        self.treedict = TreeDict(
            classification_tree=individual_treedict.classification_tree
        )

        # Sets the loss weighting for the tips
        if tip_alpha:
            for tip in self.treedict.classification_tree.leaves:
                tip.parent.alpha = tip_alpha

        self.logger.info("Loading memmap")
        self.accession_to_array_index = defaultdict(list)
        with open(memmap_index) as f:
            for key_index, key in enumerate(f):
                key = key.strip()
                accession = key.strip().split("/")[0]

                if len(self.accession_to_array_index[accession]) == 0:
                    self.treedict[accession] = individual_treedict[key]

                self.accession_to_array_index[accession].append(key_index)
        count = key_index + 1
        self.array = read_memmap(memmap, count)

        # If there's enough memory, then read into RAM
        if in_memory:
            self.array = np.array(self.array)

        self.classification_tree = self.treedict.classification_tree
        assert self.classification_tree is not None

        # Get list of gene families
        family_ids = set()
        for accession in self.treedict:
            gene_id = accession.split("/")[-1]
            family_ids.add(gene_id)

    @method
    def model(
        self,
        features: int = 768,
        intermediate_layers: int = 2,
        growth_factor: float = 2.0,
        attention_size: int = 512,
    ) -> "nn.Module":
        from barbet.models import BarbetModel

        return BarbetModel(
            classification_tree=self.classification_tree,
            features=features,
            intermediate_layers=intermediate_layers,
            growth_factor=growth_factor,
            attention_size=attention_size,
        )

    @method
    def loss_function(self):
        from hierarchicalsoftmax import HierarchicalSoftmaxLoss

        return HierarchicalSoftmaxLoss(root=self.classification_tree)

    @method
    def metrics(self) -> "list[tuple[str,Metric]]":
        from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric
        from barbet.data import RANKS

        rank_accuracy = RankAccuracyTorchMetric(
            root=self.classification_tree,
            ranks={1 + i: rank for i, rank in enumerate(RANKS)},
        )

        return [("rank_accuracy", rank_accuracy)]

    @method
    def data(
        self,
        max_items: int = 0,
        num_workers: int = 4,
        validation_partition: int = 0,
        batch_size: int = 4,
        test_partition: int = -1,
        train_all: bool = False,
    ) -> "Iterable|L.LightningDataModule":
        from barbet.data import BarbetDataModule

        return BarbetDataModule(
            array=self.array,
            accession_to_array_index=self.accession_to_array_index,
            treedict=self.treedict,
            max_items=max_items,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_partition=validation_partition,
            test_partition=test_partition,
            stack_size=self.stack_size,
            train_all=train_all,
        )

    @method
    def module_class(self) :
        from .modules import BarbetLightningModule
        return BarbetLightningModule

    @method
    def extra_hyperparameters(self, embedding_model: str = "") -> dict:
        """Extra hyperparameters to save with the module."""
        assert embedding_model, "Please provide an embedding model."
        from barbet.embeddings.esm import ESMEmbedding

        embedding_model = embedding_model.lower()
        if embedding_model.startswith("esm"):
            layers = embedding_model[3:].strip()
            embedding_model = ESMEmbedding()
            embedding_model.setup(layers=layers)
        else:
            raise ValueError(f"Cannot understand embedding model: {embedding_model}")

        return dict(
            embedding_model=embedding_model,
            classification_tree=self.treedict.classification_tree,
            stack_size=self.stack_size,
        )

    @method
    def prediction_dataloader(
        self,
        module,
        genome_path: Path,
        markers: dict[str, str], 
        batch_size: int = Param(
            64, help="The batch size for the prediction dataloader."
        ),
        cpus: int = Param(
            1, param_decls=["--cpus", "-c"], help="The number of CPUs to use for the prediction dataloader."
        ),
        dataloader_workers: int = Param(
            4, help="The number of workers to use for the dataloader."
        ),
        repeats: int = Param(
            2,
            help="The minimum number of times to use each protein embedding in the prediction.",
        ),
        genome_idx: int = Param(1, hidden=True),
        total_genomes: int = Param(1, hidden=True),
        **kwargs,
    ) -> "Iterable":
        import torch
        import numpy as np
        from torch.utils.data import DataLoader
        from barbet.data import BarbetPredictionDataset

        # Set PyTorch thread limits
        torch.set_num_threads(cpus)
       
        # Get hyperparameters from checkpoint
        stack_size = module.hparams.get("stack_size", 32)
        self.classification_tree = module.hparams.classification_tree

        # extract domain from the model
        domain = "ar53" if self.classification_tree.name == "d__Archaea" else "bac120"

        #######################
        # Create Embeddings
        #######################
        embeddings = []
        accessions = []

        fastas = sorted(markers[domain])  # sort for determinism independent of HMMER output order
        pct = (genome_idx / total_genomes) * 100 if total_genomes else 100.0
        description = f"[cyan]Embedding ({genome_idx:,}/{total_genomes:,} genomes, {pct:.1f}%)..."

        embedding_model = module.hparams.embedding_model

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(description, total=len(fastas))
            for fasta in fastas:
                # read the fasta file sequence remove the header
                fasta = Path(fasta)
                seq = fasta.read_text().split("\n")[1]
                vector = embedding_model(seq)
                if vector is not None and not torch.isnan(vector).any():
                    vector = vector.cpu().detach().clone().numpy()
                    embeddings.append(vector)

                    gene_family_id = fasta.stem
                    accession = f"{genome_path.name}/{gene_family_id}"
                    accessions.append(accession)

                del vector
                progress.advance(task)

        embeddings = np.asarray(embeddings).astype(np.float16)

        self.prediction_dataset = BarbetPredictionDataset(
            array=embeddings,
            accessions=accessions,
            stack_size=stack_size,
            repeats=repeats,
            seed=42,
        )
        dataloader = DataLoader(
            self.prediction_dataset,
            batch_size=batch_size,
            num_workers=dataloader_workers,
            shuffle=False,
        )

        return dataloader

    def node_to_str(self, node: "SoftmaxNode") -> str:
        """
        Converts the node to a string
        """
        return str(node).split(",")[-1].strip()

    @main(
        "load_checkpoint",
        "prediction_trainer",
        "prediction_dataloader",
    )
    def predict(
        self,
        input: list[Path] = Param(
            default=...,
            param_decls=["--input", "-i"],
            help="FASTA files, directories of FASTA files, or a TSV file (columns: fasta_path, [genome_id], [translation_table]). Requires genomes to be in individual FASTA file."
        ),
        output_dir: Path = Param(
            default="output",
            param_decls=["--output-dir", "-o"], 
            help="A path to the output directory."
        ),
        output_csv: Path = Param(
            default=None, 
            help="A path to output the results as a CSV."
        ),
        rm_intermediate: bool = Param(
            default=False, 
            help="If set, remove the intermediate results directory after processing."
        ),
        cpus: int = Param(
            default=1, 
            param_decls=["--cpus", "-c"], 
            help="The number of CPUs to use."
        ),
        pfam_db: str = Param(
            default="https://data.ace.uq.edu.au/public/gtdbtk/release95/markers/pfam/Pfam-A.hmm",
            help="The Pfam database to use.",
        ),
        tigr_db: str = Param(
            default="https://data.ace.uq.edu.au/public/gtdbtk/release95/markers/tigrfam/tigrfam.hmm",
            help="The TIGRFAM database to use.",
        ),
        output_predictions: Path = Param(
            default=None,
            help=(
                "A path to output the top N predictions per rank as a TSV file. "
                "A bare filename is placed in --output-dir and a directory gets 'barbet-top-predictions.tsv'."
            ),
        ),
        num_predictions: int = Param(
            default=1,
            help="The number of top predictions per rank to output when --output-predictions is specified."
        ),
        global_predictions: bool = Param(
            default=False,
            help=(
                "When set, the top N taxa at each rank are chosen globally across the entire "
                "taxonomy (not just within the children of the greedy top-1 parent). "
                "Taxa that are the only child of their parent are ranked with their parent's probability. "
                "Requires --output-predictions to be set. "
                "Output columns change to joint_probability, local_probability, prediction_number, "
                "and in_predicted_lineage."
            ),
        ),
        context_vector_file: Path = Param(
            default=None,
            help=(
                "A path to output the N-dimensional mean-pooled context vectors per genome as a TSV file "
                "(gzipped if the path ends in '.gz'). A bare filename is placed in --output-dir and a "
                "directory gets 'barbet-context-vectors.tsv'."
            ),
        ),
        **kwargs,
    ):
        """Barbet is a tool for assigning taxonomic labels to genomes using Machine Learning."""
        start_time = time.perf_counter()
        self.start_time = start_time
        self.output_dir = Path(output_dir)
        self.logger = setup_logger(self.output_dir)

        # import pandas as pd
        import polars as pl
        from itertools import chain
        from barbet.markers import extract_markers_genes

        # Get list of files & metadata
        files = []
        genome_id_map = {}  # str(file_path) -> genome_id
        translation_tables = {}  # genome_id -> translation_table (int)

        if isinstance(input, (str, Path)):
            input = [Path(input)]
        else:
            input = [Path(p) for p in input]

        assert len(input) > 0, "No input files provided."

        def _is_tsv_file(p: Path) -> bool:
            if not p.is_file():
                return False
            if p.suffix.lower() in (".tsv", ".tab"):
                return True
            if p.suffix.lower() in (".fa", ".fasta", ".fna", ".gz"):
                return False
            try:
                with open(p, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                    if first_line and not first_line.startswith(">") and "\t" in first_line:
                        return True
            except Exception:
                pass
            return False

        if len(input) == 1 and _is_tsv_file(input[0]):
            tsv_path = input[0]
            with open(tsv_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#"):
                        continue
                    parts = [p.strip() for p in line_str.split("\t")]
                    if not parts or not parts[0]:
                        continue

                    fasta_str = parts[0]
                    fasta_path = Path(fasta_str)

                    # Skip header line if col1 looks like header and file doesn't exist
                    if line_idx == 1 and not fasta_path.exists():
                        col1_lower = fasta_str.lower()
                        if col1_lower in ("path", "fasta", "fasta_path", "file", "genome_path", "genome") or "fasta" in col1_lower or "path" in col1_lower:
                            continue

                    if not fasta_path.exists():
                        raise FileNotFoundError(f"FASTA file specified in TSV does not exist: {fasta_str}")

                    files.append(fasta_path)

                    col2 = parts[1] if len(parts) > 1 and parts[1] else None
                    gid = col2 if col2 else fasta_path.name
                    genome_id_map[str(fasta_path)] = gid

                    if len(parts) > 2 and parts[2]:
                        raw_tt = parts[2]
                        try:
                            tt = int(raw_tt)
                        except ValueError:
                            raise ValueError(
                                f"Invalid translation table '{raw_tt}' at line {line_idx} in {tsv_path}. Must be 4, 11, or 25."
                            )
                        if tt not in (4, 11, 25):
                            raise ValueError(
                                f"Invalid translation table '{tt}' at line {line_idx} in {tsv_path}. Must be either 4, 11, or 25."
                            )
                        translation_tables[gid] = tt
        else:
            for path in input:
                if path.is_dir():
                    for file in chain(
                        path.rglob("*.fa"),
                        path.rglob("*.fasta"),
                        path.rglob("*.fna"),
                        path.rglob("*.fa.gz"),
                        path.rglob("*.fasta.gz"),
                        path.rglob("*.fna.gz"),
                    ):
                        files.append(file)
                        genome_id_map[str(file)] = file.name
                elif path.is_file():
                    files.append(path)
                    genome_id_map[str(path)] = path.name

        # Check if any files were found
        if len(files) == 0:
            raise ValueError(
                f"No files found in {input}. Please provide a directory, a list of files, or a TSV file."
            )

        # Check if output directory exists
        output_csv = output_csv or self.output_dir / "barbet-predictions.csv"
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(exist_ok=True, parents=True)
        self.logger.info(
            f"Writing results for {len(files)} genome{'s' if len(files) > 1 else ''} to '{output_csv}'"
        )

        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True, parents=True)

        ####################
        # Extract single copy marker genes
        ####################
        start_time_markers = time.perf_counter()
        markers_gene_map = extract_markers_genes(
            genomes={genome_id_map[str(file)]: str(file) for file in files},
            out_dir=str(results_dir),
            cpus=cpus,
            force=True,
            pfam_db=self.process_location(pfam_db),
            tigr_db=self.process_location(tigr_db),
            translation_tables=translation_tables if translation_tables else None,
        )
        extract_markers_time = time.perf_counter() - start_time_markers
        self.logger.info(
            f"Total time to extract and identify marker genes: {extract_markers_time:.2f} seconds"
        )

        # Load the model
        module = self.load_checkpoint(**kwargs)
        trainer = self.prediction_trainer(module, **kwargs)

        # Make predictions for each file
        total_df = None
        total_genomes = len(markers_gene_map)
        kwargs_dataloader = dict(kwargs)
        kwargs_dataloader.pop("genome_idx", None)
        kwargs_dataloader.pop("total_genomes", None)

        stack_size = module.hparams.get("stack_size", 32)
        repeats = kwargs.get("repeats", 2)
        batch_size = kwargs.get("batch_size", 64)
        dataloader_workers = kwargs.get("dataloader_workers", 4)

        start_time_embed_classify = time.perf_counter()

        all_embeddings_list = []
        all_accessions = []

        for idx, (gid, maker_genes) in enumerate(markers_gene_map.items(), start=1):
            genome_path = Path(gid)
            self.prediction_dataloader(
                module,
                genome_path,
                maker_genes,
                cpus=cpus,
                genome_idx=idx,
                total_genomes=total_genomes,
                **kwargs_dataloader,
            )
            if len(self.prediction_dataset.array) > 0:
                all_embeddings_list.append(self.prediction_dataset.array)
                all_accessions.extend(self.prediction_dataset.accessions)

        if all_embeddings_list:
            import numpy as np
            from torch.utils.data import DataLoader
            from barbet.data import BarbetPredictionDataset

            all_embeddings_arr = np.concatenate(all_embeddings_list, axis=0)
            self.prediction_dataset = BarbetPredictionDataset(
                array=all_embeddings_arr,
                accessions=all_accessions,
                stack_size=stack_size,
                repeats=repeats,
                seed=42,
            )
            combined_dataloader = DataLoader(
                self.prediction_dataset,
                batch_size=batch_size,
                num_workers=dataloader_workers,
                shuffle=False,
            )
            names = [stack.genome for stack in self.prediction_dataset.stacks]
            # Top-N predictions need the full probability matrix to be retained after
            # inference; this does not change how barbet-predictions.csv is calculated.
            save_ctx = bool(context_vector_file)
            if global_predictions and not output_predictions:
                self.logger.warning(
                    "--global-predictions has no effect without --output-predictions."
                )
            module.setup_prediction(
                self,
                names,
                retain_probabilities=bool(output_predictions),
                save_context_vectors=save_ctx,
            )
            trainer.predict(module, dataloaders=combined_dataloader)
            total_df = module.results_df

            # Write the main results before the optional outputs so that they are kept if one of these fails
            if output_csv:
                total_df.write_csv(output_csv)
                self.logger.info(f"Saved to: '{output_csv}'")

            if output_predictions:
                out_pred_path = resolve_output_path(output_predictions, self.output_dir, "barbet-top-predictions.tsv")
                top_preds_df = module.generate_top_predictions(
                    self,
                    num_predictions=num_predictions,
                    global_predictions=global_predictions,
                )
                top_preds_df.write_csv(out_pred_path, separator="\t")
                self.logger.info(f"Saved top predictions to: '{out_pred_path}'")

            if context_vector_file:
                out_ctx_path = resolve_output_path(context_vector_file, self.output_dir, "barbet-context-vectors.tsv")
                module.save_context_vectors_tsv(out_ctx_path)
                self.logger.info(f"Saved context vectors to: '{out_ctx_path}'")

        embed_classify_time = time.perf_counter() - start_time_embed_classify
        self.logger.info(
            f"Total time to embed and classify: {embed_classify_time:.2f} seconds"
        )

        if total_df is not None:
            print_polars_df(
                total_df[["name", "species_prediction", "species_probability", ]],
                column_names=["Genome", "Species", "Probability"],
            )
        else:
            self.logger.warning("No predictions were generated.")

        if rm_intermediate and results_dir.exists():
            import shutil
            shutil.rmtree(results_dir)

        total_time = time.perf_counter() - start_time
        self.logger.info(f"Total time: {total_time:.2f} seconds")
        self.logger.info("Done")
        return total_df

    @tool(
        "load_checkpoint",
        "prediction_trainer",
        "prediction_dataloader_memmap",
    )
    def predict_memmap(
        self,
        output_csv: Path = Param(
            default=None, help="A path to output the results as a CSV."
        ),
        treedict:Path = Param(None, help="A path to a TreeDict with the ground truth lineage."),
        probabilities: bool = Param(
            default=False, help="If True, include probabilities for all the nodes in the taxonomic tree."
        ),
        context_vector_file: Path = Param(
            default=None,
            help=(
                "A path to output the N-dimensional mean-pooled context vectors per genome as a TSV file "
                "(gzipped if the path ends in '.gz')."
            ),
        ),
        **kwargs,
    ):
        """Barbet is a tool for assigning taxonomic labels to genomes using Machine Learning."""
        start_time = time.perf_counter()
        self.start_time = start_time
        module = self.load_checkpoint(**kwargs)
        trainer = self.prediction_trainer(module, **kwargs)
        prediction_dataloader = self.prediction_dataloader_memmap(module, **kwargs)

        save_ctx = bool(context_vector_file)
        module.setup_prediction(
            self,
            [stack.genome for stack in self.prediction_dataset.stacks],
            save_probabilities=probabilities,
            save_context_vectors=save_ctx,
        )
        trainer.predict(module, dataloaders=prediction_dataloader, return_predictions=False)
        results_df = module.results_df

        genome_name_set = set(results_df['name'].unique())

        if treedict is not None:
            from hierarchicalsoftmax import TreeDict
            from barbet.data import RANKS
            import polars as pl

            true_values = defaultdict(dict)

            self.logger.info(f"Adding true values from TreeDict '{treedict}'")
            treedict = TreeDict.load(treedict)
            
            # Get lineage to map
            for accession in track(treedict.keys()):
                genome_name = accession.split("/")[0]
                if genome_name in genome_name_set:
                    node = treedict.node(accession)
                    lineage = node.ancestors[1:] + (node,)
                    for rank, lineage_node in zip(RANKS, lineage):
                        true_values[rank][genome_name] = lineage_node.name.strip()


            for rank in RANKS:
                results_df = results_df.with_columns(
                    pl.col("name").map_elements(true_values[rank].get, return_dtype=pl.Utf8).alias(f"{rank}_true")
                )
    
        self.logger.info(f"Writing to '{output_csv}'")
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(exist_ok=True, parents=True)
        results_df.write_csv(output_csv)

        if context_vector_file:
            out_ctx_path = resolve_output_path(
                context_vector_file, getattr(self, "output_dir", Path("output")), "barbet-context-vectors.tsv"
            )
            module.save_context_vectors_tsv(out_ctx_path)
            self.logger.info(f"Saved context vectors to: '{out_ctx_path}'")

        total_time = time.perf_counter() - start_time
        self.logger.info(f"Total time: {total_time:.2f} seconds")
        self.logger.info("Done")

        return results_df
    
    @method
    def prediction_dataloader_memmap(
        self,
        module,
        memmap:Path = Param(None, help="A path to the memmap file containing the protein embeddings."),
        memmap_index:Path = Param(None, help="A path to the memmap index file containing the accessions."),
        batch_size: int = Param(
            64, help="The batch size for the prediction dataloader."
        ),
        num_workers: int = 4,
        repeats: int = Param(
            2,
            help="The minimum number of times to use each protein embedding in the prediction.",
        ),
        genomes:Path=Param(None, help="A path to a text file with the accessions for the genome to use."),
        **kwargs,
    ) -> "Iterable":
        from barbet.data import read_memmap
        from torch.utils.data import DataLoader
        from barbet.data import BarbetPredictionDataset
        
        assert memmap is not None, "Please provide a path to the memmap file."
        assert memmap.exists(), f"Memmap file does not exist: {memmap}"
        assert memmap_index is not None, "Please provide a path to the memmap index file."
        assert memmap_index.exists(), f"Memmap index file does not exist: {memmap_index}"

        # Read the memmap array index
        self.logger.info(f"Reading memmap array index '{memmap_index}'")
        accessions = memmap_index.read_text().strip().split("\n")
        count = len(accessions)
        self.logger.info(f"Found {count} accessions")

        # Load the memmap array itself
        self.logger.info(f"Loading memmap array '{memmap}'")
        array = read_memmap(memmap, count)

        # Get hyperparameters from checkpoint
        self.classification_tree = module.hparams.classification_tree
        stack_size = module.hparams.get("stack_size", 32)

        # If treedict is provided, then we filter the accessions to only those that are in the treedict
        genome_filter = None
        if genomes:
            assert genomes.exists(), f"Genomes file does not exist: {genomes}"
            genome_filter = set(Path(genomes).read_text().strip().split("\n"))

        self.prediction_dataset = BarbetPredictionDataset(
            array=array,
            accessions=accessions,
            stack_size=stack_size,
            repeats=repeats,
            genome_filter=genome_filter,
            seed=42,
        )
        dataloader = DataLoader(
            self.prediction_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        return dataloader

    @method
    def monitor(
        self,
        train_all: bool = False,
        **kwargs,
    ) -> str:
        if train_all:
            return "valid_loss"
        return "genus"

    def checkpoint(
        self, 
        checkpoint:Path=Param(None, help="The path to a checkpoint file for the Barbet parameters. If not provided, then it will use a standard checkpoint."), 
        large:bool=Param(False, help="Whether or not to use the large standard checkpoint of the Barbet parameters."), 
        archaea:bool=Param(False, help="Whether or not to use the standard model for archaea. If not, then it uses the default model for bacteria."),
    ) -> str:
        if checkpoint:
            return checkpoint
        
        # Weights are here: https://figshare.unimelb.edu.au/articles/dataset/Trained_weights_for_Barbet/
        # DOI: https://doi.org/10.26188/29578964

        if archaea:
            if large: 
                # barbet-ar53-ESM12-large.ckpt
                return "https://figshare.unimelb.edu.au/ndownloader/files/56332160"
            
            # barbet-ar53-ESM12-base.ckpt
            return "https://figshare.unimelb.edu.au/ndownloader/files/56332157"
        
        if large:
            # barbet-bac120-ESM6-large.ckpt
            return "https://figshare.unimelb.edu.au/ndownloader/files/56307647"
        
        # barbet-bac120-ESM6-base.ckpt
        return "https://figshare.unimelb.edu.au/ndownloader/files/56307671"
