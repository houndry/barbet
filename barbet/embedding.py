import gzip
import os
from pathlib import Path
from abc import ABC, abstractmethod
from Bio import SeqIO
import random
import numpy as np
from rich.progress import track
from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax import TreeDict
import tarfile
import torch
from io import StringIO
from torchapp.cli import CLIApp, tool, method
import typer
from dataclasses import dataclass

from .data import read_memmap, RANKS


def _open(path, mode='rt', **kwargs):
    """
    Open a file normally, or with gzip if it ends in .gz.
    
    Args:
        path (str or Path): The path to the file.
        mode (str): The mode to open the file with (default 'rt' for reading text).
        **kwargs: Additional arguments passed to open or gzip.open.

    Returns:
        A file object.
    """
    path = Path(path)
    if path.suffix == '.gz':
        return gzip.open(path, mode, **kwargs)
    return open(path, mode, **kwargs)


def set_validation_rank_to_treedict(
    treedict:TreeDict,
    validation_rank:str="species",
    partitions:int=5,
) -> TreeDict:
    # find the taxonomic rank to use for the validation partition
    validation_rank = validation_rank.lower()
    assert validation_rank in RANKS
    validation_rank_index = RANKS.index(validation_rank)

    partitions_dict = {}
    for key in treedict:
        node = treedict.node(key)             
        # Assign validation partition at set rank
        partition_node = node.ancestors[validation_rank_index]
        if partition_node not in partitions_dict:
            partitions_dict[partition_node] = random.randint(0,partitions-1)

        treedict[key].partition = partitions_dict[partition_node]

    return treedict


def get_key(accession:str, gene:str) -> str:
    """ Returns the standard format of a key """
    key = f"{accession}/{gene}"
    return key


def get_node(lineage:str, lineage_to_node:dict[str,SoftmaxNode]) -> SoftmaxNode:
    if lineage in lineage_to_node:
        return lineage_to_node[lineage]

    assert ";" in lineage, f"Semi-colon ';' not found in lineage '{lineage}'"
    split_point = lineage.rfind(";")
    parent_lineage = lineage[:split_point]
    name = lineage[split_point+1:]
    parent = get_node(parent_lineage, lineage_to_node)
    node = SoftmaxNode(name=name, parent=parent)
    lineage_to_node[lineage] = node
    return node


def generate_overlapping_intervals(total: int, interval_size: int, min_overlap: int, check:bool=True, variable_size:bool=False) -> list[tuple[int, int]]:
    """
    Creates a list of overlapping intervals within a specified range, adjusting the interval size to ensure
    that the overlap is approximately the same across all intervals.

    Args:
        total (int): The total range within which intervals are to be created.
        max_interval_size (int): The maximum size of each interval.
        min_overlap (int): The minimum number of units by which consecutive intervals overlap.
        check (bool): If True, checks are performed to ensure that the intervals meet the specified conditions.

    Returns:
        list[tuple[int, int]]: A list of tuples where each tuple represents the start (inclusive) 
        and end (exclusive) of an interval.

    Example:
        >>> generate_overlapping_intervals(20, 5, 2)
        [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17), (15, 20)]
    """
    intervals = []
    start = 0

    if total == 0:
        return intervals
    
    max_interval_size = interval_size
    assert interval_size
    assert min_overlap is not None
    assert interval_size > min_overlap, f"Max interval size of {interval_size} must be greater than min overlap of {min_overlap}"

    # Calculate the number of intervals needed to cover the range
    num_intervals, remainder = divmod(total - min_overlap, interval_size - min_overlap)
    if remainder > 0:
        num_intervals += 1

    # Calculate the exact interval size to ensure consistent overlap
    overlap = min_overlap
    if variable_size:
        if num_intervals > 1:
            interval_size, remainder = divmod(total + (num_intervals - 1) * overlap, num_intervals)
            if remainder > 0:
                interval_size += 1
    else:
        # If the size is fixed, then vary the overlap to keep it even
        if num_intervals > 1:
            overlap, remainder = divmod( num_intervals * interval_size - total, num_intervals - 1)
            if overlap < min_overlap:
                overlap = min_overlap

    while True:
        end = start + interval_size
        if end > total:
            end = total
            start = max(end - interval_size,0)
        intervals.append((start, end))
        start += interval_size - overlap
        if end >= total:
            break

    if check:
        assert intervals[0][0] == 0
        assert intervals[-1][1] == total
        assert len(intervals) == num_intervals, f"Expected {num_intervals} intervals, got {len(intervals)}"

        assert interval_size <= max_interval_size, f"Interval size of {interval_size} exceeds max interval size of {max_interval_size}"
        for interval in intervals:
            assert interval[1] - interval[0] == interval_size, f"Interval size of {interval[1] - interval[0]} is not the expected size {interval_size}"

        for i in range(1, len(intervals)):
            overlap = intervals[i - 1][1] - intervals[i][0]
            assert overlap >= min_overlap, f"Min overlap condition of {min_overlap} not met for intervals {intervals[i - 1]} and {intervals[i]} (overlap {overlap})"

    return intervals


@dataclass
class Embedding(CLIApp, ABC):
    """ A class for embedding protein sequences. """
    max_length:int|None=None
    overlap:int=64
    mean_pool:bool=True

    def __post_init__(self):
        super().__init__()

    @abstractmethod
    def embed(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding vector. """
        raise NotImplementedError

    def reduce(self, tensor:torch.Tensor) -> torch.Tensor:
        if not self.mean_pool:
            return tensor
        
        if tensor.ndim == 2:
            tensor = tensor.mean(dim=0)
        assert tensor.ndim == 1
        return tensor

    def __call__(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding vector. """
        if not self.max_length or len(seq) <= self.max_length:
            tensor = self.embed(seq)
            return self.reduce(tensor)
        
        epsilon = 0.1
        intervals = generate_overlapping_intervals(len(seq), self.max_length, self.overlap)
        weights = torch.zeros( (len(seq),), device="cpu" )
        tensor = None
        for start,end in intervals:
            result = self.embed(seq[start:end]).cpu()

            assert result.shape[0] == end-start
            embedding_size = result.shape[1]

            if tensor is None:
                tensor = torch.zeros( (len(seq), embedding_size ), device="cpu")

            assert tensor.shape[-1] == embedding_size

            interval_indexes = torch.arange(end-start)
            distance_from_ends = torch.min( interval_indexes-start, end-interval_indexes-1 )
            
            weight = epsilon + torch.minimum(distance_from_ends, torch.tensor(self.overlap))

            tensor[start:end] += result * weight.unsqueeze(1)
            weights[start:end] += weight

        tensor = tensor/weights.unsqueeze(1)

        return self.reduce(tensor)
    
    @method
    def setup(self, **kwargs):
        pass

    def build_treedict(self, taxonomy:Path) -> tuple[TreeDict,dict[str,SoftmaxNode]]:
        # Create root of tree
        lineage_to_node = {}
        root = None

        # Fill out tree with taxonomy
        accession_to_node = {}
        with _open(taxonomy) as f:
            for line in f:
                accesssion, lineage = line.split("\t")

                if not root:
                    root_name = lineage.split(";")[0]
                    root = SoftmaxNode(root_name)
                    lineage_to_node[root_name] = root

                node = get_node(lineage, lineage_to_node)
                accession_to_node[accesssion] = node
        
        treedict = TreeDict(classification_tree=root)
        return treedict, accession_to_node

    @tool("setup")
    def test_lengths(
        self,
        end:int=5_000,
        start:int=1000,
        retries:int=5,
        **kwargs,
    ):
        def random_amino_acid_sequence(k):
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # standard 20 amino acids
            return ''.join(random.choice(amino_acids) for _ in range(k))
        
        self.max_length = None
        self.setup(**kwargs)
        for ii in track(range(start,end)):
            for _ in range(retries):
                seq = random_amino_acid_sequence(ii)
                try:
                    self(seq)
                except Exception as err:
                    print(f"{ii}: {err}")
                    return


    @tool("setup")
    def build_gene_array(
        self,
        marker_genes:Path=typer.Option(default=..., help="The path to the marker genes tarball (e.g. bac120_msa_marker_genes_all_r220.tar.gz)."),
        family_index:int=typer.Option(default=..., help="The index for the gene family to use. E.g. if there are 120 gene families then this should be a number from 0 to 119."),
        output_dir:Path=typer.Option(default=..., help="A directory to store the output which includes the memmap array, the listing of accessions and an error log."),
        flush_every:int=typer.Option(default=5_000, help="An interval to flush the memmap array as it is generated."),
        max_length:int=None,
        **kwargs,
    ):
        self.max_length = max_length
        self.setup(**kwargs)

        assert marker_genes is not None
        assert family_index is not None
        assert output_dir is not None

        dtype = 'float16'

        memmap_wip_array = None
        output_dir.mkdir(parents=True, exist_ok=True)
        memmap_wip_path = output_dir / f"{family_index}-wip.npy"
        error = output_dir / f"{family_index}-errors.txt"
        accessions_wip = output_dir / f"{family_index}-accessions-wip.txt"

        accessions = []
        
        print(f"Loading {marker_genes} file.")
        with tarfile.open(marker_genes, "r:gz") as tar, open(error, "w") as error_file, open(accessions_wip, "w") as accessions_wip_file:
            members = [member for member in tar.getmembers() if member.isfile() and member.name.endswith(".faa")]
            prefix_length = len(os.path.commonprefix([Path(member.name).with_suffix("").name for member in members]))
            
            member = members[family_index]
            print(f"Processing file {family_index} in {marker_genes}")

            f = tar.extractfile(member)
            marker_id = Path(member.name).with_suffix("").name[prefix_length:]

            fasta_io = StringIO(f.read().decode('ascii'))

            total = sum(1 for _ in SeqIO.parse(fasta_io, "fasta"))
            fasta_io.seek(0)
            print(marker_id, total)
    
            for record in track(SeqIO.parse(fasta_io, "fasta"), total=total):
            # for record in SeqIO.parse(fasta_io, "fasta"):
                species_accession = record.id
                                
                key = get_key(species_accession, marker_id)

                seq = str(record.seq).replace("-","").replace("*","")
                try:
                    vector = self(seq)
                except Exception as err:
                    print(f"{key} ({len(seq)}): {err}", file=error_file)
                    print(f"{key} ({len(seq)}): {err}")
                    continue

                if vector is None:
                    print(f"{key} ({len(seq)}): Embedding is None", file=error_file)
                    print(f"{key} ({len(seq)}): Embedding is None")
                    continue

                if torch.isnan(vector).any():
                    print(f"{key} ({len(seq)}): Embedding contains NaN", file=error_file)
                    print(f"{key} ({len(seq)}): Embedding contains NaN")
                    continue

                if memmap_wip_array is None:
                    size = len(vector)
                    shape = (total,size)
                    memmap_wip_array = np.memmap(memmap_wip_path, dtype=dtype, mode='w+', shape=shape)

                index = len(accessions)
                memmap_wip_array[index,:] = vector.cpu().half().numpy()
                if index % flush_every == 0:
                    memmap_wip_array.flush()
                
                accessions.append(key)
                print(key, file=accessions_wip_file)
                                            
        memmap_wip_array.flush()

        accessions_path = output_dir / f"{family_index}.txt"
        with open(accessions_path, "w") as f:
            for accession in accessions:
                print(accession, file=f)
        
        # Save final memmap array now that we now the final size
        memmap_path = output_dir / f"{family_index}.npy"
        shape = (len(accessions),size)
        print(f"Writing final memmap array of shape {shape}: {memmap_path}")
        memmap_array = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=shape)
        memmap_array[:len(accessions),:] = memmap_wip_array[:len(accessions),:]
        memmap_array.flush()

        # Clean up
        memmap_array._mmap.close()
        memmap_array._mmap = None
        memmap_array = None
        memmap_wip_path.unlink()
        accessions_wip.unlink()

    @tool
    def set_validation_rank(
        self,
        treedict:Path=typer.Option(default=..., help="The path to the treedict file."),
        output:Path=typer.Option(default=..., help="The path to save the adapted treedict file."),
        validation_rank:str=typer.Option(default="species", help="The rank to hold out for cross-validation."),
        partitions:int=typer.Option(default=5, help="The number of cross-validation partitions."),
    ) -> TreeDict:
        treedict = TreeDict.load(treedict)
        set_validation_rank_to_treedict(treedict, validation_rank=validation_rank, partitions=partitions)
        treedict.save(output)
        return treedict

    @tool
    def preprocess(
        self,
        taxonomy:Path=typer.Option(default=..., help="The path to the TSV taxonomy file (e.g. bac120_taxonomy_r220.tsv)."),
        marker_genes:Path=typer.Option(default=..., help="The path to the marker genes tarball (e.g. bac120_msa_marker_genes_all_r220.tar.gz)."),
        output_dir:Path=typer.Option(default=..., help="A directory to store the output which includes the memmap array, the listing of accessions and an error log."),
        partitions:int=typer.Option(default=5, help="The number of cross-validation partitions."),
        seed:int=typer.Option(default=42, help="The random seed."),
        treedict_only:bool=typer.Option(default=False, help="Only output TreeDict file and then exit before concatenating memmap array"),
    ):
        treedict, accession_to_node = self.build_treedict(taxonomy)

        dtype = 'float16'

        random.seed(seed)

        print(f"Loading {marker_genes} file.")
        with tarfile.open(marker_genes, "r:gz") as tar:
            members = [member for member in tar.getmembers() if member.isfile() and member.name.endswith(".faa")]
            family_count = len(members)
        print(f"{family_count} gene families found.")

        # Read and collect accessions
        print(f"Building treedict")
        keys = []
        counts = []
        node_to_partition_dict = dict()
        for family_index in track(range(family_count)):
            keys_path = output_dir / f"{family_index}.txt"

            if not keys_path.exists():
                counts.append(0)
                continue

            with open(keys_path) as f:
                family_index_keys = [line.strip() for line in f]
                keys += family_index_keys
                counts.append(len(family_index_keys))

                for key in family_index_keys:
                    genome_accession = key.split("/")[0]
                    node = accession_to_node[genome_accession]
                    partition = node_to_partition_dict.setdefault(node, random.randint(0, partitions - 1))

                    # Add to treedict
                    treedict.add(key, node, partition)
        
        assert len(counts) == family_count

        # Save treedict
        treedict_path = output_dir / f"{output_dir.name}.td"
        print(f"Saving TreeDict to {treedict_path}")
        treedict.save(treedict_path)

        if treedict_only:
            return

        # Concatenate numpy memmap arrays
        memmap_array = None
        memmap_array_path = output_dir / f"{output_dir.name}.npy"
        print(f"Saving memmap to {memmap_array_path}")
        current_index = 0
        for family_index, family_count in track(enumerate(counts), total=len(counts)):
            my_memmap_path = output_dir / f"{family_index}.npy"

            # Build memmap for gene family if it doesn't exist
            if not my_memmap_path.exists():
                continue
                # print("Building", my_memmap_path)
                # self.build_gene_array(marker_genes=marker_genes, family_index=family_index, output_dir=output_dir)
                # assert my_memmap_path.exists()

            my_memmap = read_memmap(my_memmap_path, family_count)

            # Build memmap for output if it doesn't exist
            if memmap_array is None:
                size = my_memmap.shape[1]
                shape = (len(keys),size)
                memmap_array = np.memmap(memmap_array_path, dtype=dtype, mode='w+', shape=shape)

            # Copy memmap for gene family into output memmap
            memmap_array[current_index:current_index+family_count,:] = my_memmap[:,:]

            current_index += family_count

        assert len(keys) == current_index

        memmap_array.flush()

        # Save keys
        keys_path = output_dir / f"{output_dir.name}.txt"
        print(f"Saving keys to {keys_path}")        
        with open(keys_path, "w") as f:
            for key in keys:
                print(key, file=f)

    @tool
    def prune_to_representatives(treedict:Path, representatives:Path, output:Path):
        print("Getting list of representatives from", representatives)
        keys_to_keep = []
        with tarfile.open(representatives, "r:gz") as tar:
            members = [member for member in tar.getmembers() if member.isfile() and member.name.endswith(".faa")]
            
            print(f"Processing {len(members)} files in {representatives}")

            for member in track(members):
                f = tar.extractfile(member)
                marker_id = Path(member.name.split("_")[-1]).with_suffix("").name

                fasta_io = StringIO(f.read().decode('ascii'))

                for record in SeqIO.parse(fasta_io, "fasta"):
                    species_accession = record.id
                    key = get_key(species_accession, marker_id)
                    keys_to_keep.append(key)

        # keys_to_keep = set(keys_to_keep)
        print(f"Keeping {len(keys_to_keep)} representatives")

        print(f"Loading treedict {treedict}")
        
        treedict = TreeDict.load(treedict)
        print("Total", len(treedict))
        missing = []
        for key in track(keys_to_keep):
            if key not in treedict:
                missing.append(key)

        print(f"{len(missing)} representatives missing output {len(keys_to_keep)} (total: {len(treedict)})")
        if len(missing):
            keys_to_keep = [k for k in keys_to_keep if k not in missing]

        new_treedict = TreeDict(treedict.classification_tree)
        new_treedict.update({k:treedict[k] for k in keys_to_keep})
        print("Total after pruning", len(new_treedict))

        print("Saving treedict to", output)
        new_treedict.save(output)        
