import pytest
from pathlib import Path
import torch
import polars as pl
from hierarchicalsoftmax import SoftmaxNode

from barbet import Barbet
from barbet.embedding import Embedding
from barbet.modules import BarbetLightningModule
from barbet.data import RANKS


TEST_DATA_DIR = Path(__file__).parent / "data"


class MockPredictionTrainer():
    def predict(self, module, dataloaders):
        result = [
            torch.zeros( (32, 5) ),
            torch.zeros( (32, 5) ),
        ]
        result[0][:, -1] = 1
        return result


class MockEmbedding(Embedding):
    def embed(self, seq: str) -> str:
        embedding = torch.zeros((len(seq), 26))
        embedding[0, -1] = 1
        return embedding


class MockHParams:
    def __init__(self):
        self.embedding_model = MockEmbedding()
        root = SoftmaxNode('root')
        SoftmaxNode('A', parent=root)
        SoftmaxNode('B', parent=root)
        SoftmaxNode('C', parent=root)
        SoftmaxNode('D', parent=root)
        SoftmaxNode('E', parent=root)
        root.set_indexes()
        self.classification_tree = root

    def get(self, key, default=None):
        return getattr(self, key, default)


class MockCheckpoint():
    def __init__(self, *args, **kwargs):
        pass
        
    @property
    def hparams(self):
        return MockHParams()

    def setup_prediction(
        self,
        barbet,
        names: list[str] | str,
        threshold: float = 0.0,
        save_probabilities: bool = False,
        save_context_vectors: bool = False,
        retain_probabilities: bool = False,
    ):
        if isinstance(names, str):
            names = [names]
        unique_names = list(dict.fromkeys(names))
        self.name_to_index = {name: i for i, name in enumerate(unique_names)}
        self.classification_tree = self.hparams.classification_tree
        non_root_nodes = [node for node in self.classification_tree.node_list_softmax if not node.is_root]
        self.probabilities = torch.ones((len(unique_names), len(non_root_nodes))) * 0.2
        self.save_context_vectors = save_context_vectors
        if save_context_vectors:
            self.context_vectors = torch.ones((len(unique_names), 3072), dtype=torch.float32) * 0.5
        self.results_df = pl.DataFrame(
            {
                "name": unique_names,
                "species_probability": [0.295] * len(unique_names),
                "species_prediction": ["E"] * len(unique_names),
            }
        )

    def generate_top_predictions(self, barbet, num_predictions: int = 1, global_predictions: bool = False):
        return BarbetLightningModule.generate_top_predictions(self, barbet, num_predictions=num_predictions, global_predictions=global_predictions)

    def save_context_vectors_tsv(self, output_path: str | Path) -> None:
        BarbetLightningModule.save_context_vectors_tsv(self, output_path)


@pytest.mark.parametrize("k", [1,2])
def test_predict(k, tmp_path):
    barbet = Barbet()
    # mock checkpoint
    barbet.load_checkpoint = lambda *args, **kwargs: MockCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()

    output_dir = tmp_path / "output"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # copy test data to temp directory
    input = []
    for i, file in enumerate([TEST_DATA_DIR / "MAG-GUT41.fa.gz"] * k):
        output_file = tmp_path / f"{i}.fa.gz"
        output_file.write_bytes(file.read_bytes())
        input.append(output_file)

    results = barbet.predict(input=input, output_dir=output_dir, image_format="dot")
    
    # Check output directory
    assert output_dir.exists()    
    assert (output_dir / "barbet.log").exists()
    assert (output_dir / "barbet-predictions.csv").exists()
    assert (output_dir / "results" / "0.fa.gz").exists()
    assert (output_dir / "results" / "0.fa.gz" / "pfam.tblout").exists()

    # Check result df
    assert len(results) == k
    assert 'name' in results.columns
    assert 'species_prediction' in results.columns
    assert 'species_probability' in results.columns
    for i, row in enumerate(results.iter_rows(named=True)):
        assert f"{i}.fa.gz" in row['name']
        assert row['species_prediction'] == 'E'
        assert 0.29 < row['species_probability'] < 0.30


def test_predict_rm_intermediate(tmp_path):
    barbet = Barbet()
    barbet.load_checkpoint = lambda *args, **kwargs: MockCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()

    output_dir = tmp_path / "output_rm"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    input_file = tmp_path / "0.fa.gz"
    input_file.write_bytes((TEST_DATA_DIR / "MAG-GUT41.fa.gz").read_bytes())

    results = barbet.predict(input=[input_file], output_dir=output_dir, rm_intermediate=True)

    assert output_dir.exists()
    assert (output_dir / "barbet-predictions.csv").exists()
    assert (output_dir / "barbet.log").exists()
    assert not (output_dir / "results").exists()


def test_predict_tsv_input(tmp_path):
    barbet = Barbet()
    barbet.load_checkpoint = lambda *args, **kwargs: MockCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()

    output_dir = tmp_path / "output_tsv"
    data_file = tmp_path / "genome.fa.gz"
    data_file.write_bytes((TEST_DATA_DIR / "MAG-GUT41.fa.gz").read_bytes())

    tsv_file = tmp_path / "input.tsv"
    tsv_file.write_text(f"{data_file}\tCustomGenome1\t11\n")

    results = barbet.predict(input=[tsv_file], output_dir=output_dir)

    assert output_dir.exists()
    assert (output_dir / "barbet-predictions.csv").exists()
    assert len(results) == 1
    assert results["name"][0] == "CustomGenome1"


def test_predict_tsv_invalid_translation_table(tmp_path):
    barbet = Barbet()
    barbet.load_checkpoint = lambda *args, **kwargs: MockCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()

    output_dir = tmp_path / "output_invalid"
    data_file = tmp_path / "genome.fa.gz"
    data_file.write_bytes((TEST_DATA_DIR / "MAG-GUT41.fa.gz").read_bytes())

    tsv_file = tmp_path / "input_invalid.tsv"
    tsv_file.write_text(f"{data_file}\tCustomGenome1\t5\n")

    with pytest.raises(ValueError, match="translation table"):
        barbet.predict(input=[tsv_file], output_dir=output_dir)


def test_cli_help_flag():
    barbet = Barbet()
    assert "-h" in barbet.main_app.info.context_settings["help_option_names"]
    assert "-h" in barbet.tools_app.info.context_settings["help_option_names"]


def test_predict_output_predictions(tmp_path):
    barbet = Barbet()
    barbet.load_checkpoint = lambda *args, **kwargs: MockCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()

    output_dir = tmp_path / "output_preds"
    data_file = tmp_path / "genome.fa.gz"
    data_file.write_bytes((TEST_DATA_DIR / "MAG-GUT41.fa.gz").read_bytes())
    out_preds_tsv = output_dir / "top_predictions.tsv"

    results = barbet.predict(
        input=[data_file],
        output_dir=output_dir,
        output_predictions=out_preds_tsv,
        num_predictions=2,
    )

    assert out_preds_tsv.exists()
    df_preds = pl.read_csv(out_preds_tsv, separator="\t")
    assert list(df_preds.columns) == ["name", "rank", "taxon", "probability", "prediction_number"]
    assert set(df_preds["prediction_number"].unique().to_list()).issubset({1, 2})
    # Verify main results dataframe / CSV does not contain extra node category columns
    assert "A" not in results.columns
    assert "B" not in results.columns


def test_predict_output_predictions_less_than_n(tmp_path):
    barbet = Barbet()
    barbet.load_checkpoint = lambda *args, **kwargs: MockCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()

    output_dir = tmp_path / "output_preds_less"
    data_file = tmp_path / "genome.fa.gz"
    data_file.write_bytes((TEST_DATA_DIR / "MAG-GUT41.fa.gz").read_bytes())
    out_preds_tsv = output_dir / "top_predictions.tsv"

    barbet.predict(
        input=[data_file],
        output_dir=output_dir,
        output_predictions=out_preds_tsv,
        num_predictions=10,
    )

    assert out_preds_tsv.exists()
    df_preds = pl.read_csv(out_preds_tsv, separator="\t")
    # Even though num_predictions=10, MockCheckpoint only has 5 candidate taxa at phylum rank
    assert len(df_preds) == 5
    assert df_preds["prediction_number"].to_list() == [1, 2, 3, 4, 5]


def test_generate_top_predictions_single_child_regression(tmp_path):
    """Regression test: generate_top_predictions must not stop early when the
    greedy top-1 node at a given rank has exactly one child that is excluded
    from node_list_softmax.

    hierarchicalsoftmax omits single-child nodes from node_list_softmax because
    no probability split is needed when there is only one option.  The old code
    treated empty valid_children as 'no more ranks to report' and broke out of
    the loop, silently dropping all deeper ranks.

    Tree structure:
        root
        ├── A   (depth 1, has 1 child → child NOT in softmax)
        │   └── A1  (depth 2, single child → not in softmax)
        └── B   (depth 1, has 2 children → both in softmax)
            ├── B1  (depth 2)
            └── B2  (depth 2)

    When A has the highest probability at depth 1, the greedy path goes A → A1.
    A1 is not in node_list_softmax, so the old code broke at depth 2.
    The fix must include A1 (inheriting A's probability) and continue.
    """
    from barbet.modules import BarbetLightningModule

    # Build the mock tree
    root = SoftmaxNode("root")
    node_a = SoftmaxNode("A", parent=root)    # depth 1, single child below
    node_a1 = SoftmaxNode("A1", parent=node_a)  # depth 2, single child → NOT in softmax
    node_b = SoftmaxNode("B", parent=root)    # depth 1, two children below
    node_b1 = SoftmaxNode("B1", parent=node_b)
    node_b2 = SoftmaxNode("B2", parent=node_b)
    root.set_indexes()

    # Verify the tree structure matches the expected softmax membership
    softmax_nodes = set(root.node_list_softmax)
    assert node_a in softmax_nodes, "A should be in softmax (has sibling B)"
    assert node_b in softmax_nodes, "B should be in softmax (has sibling A)"
    assert node_a1 not in softmax_nodes, "A1 should NOT be in softmax (single child of A)"
    assert node_b1 in softmax_nodes, "B1 should be in softmax (has sibling B2)"
    assert node_b2 in softmax_nodes, "B2 should be in softmax (has sibling B1)"

    class MockBarbet2:
        def node_to_str(self, node):
            return node.name

    barbet2 = MockBarbet2()

    # Build a mock module: A has the highest probability at depth 1
    non_root = [n for n in root.node_list_softmax if not n.is_root]
    node_idx = {n: i for i, n in enumerate(non_root)}

    import torch as _torch
    probs = _torch.zeros((1, len(non_root)))
    probs[0, node_idx[node_a]] = 0.9   # A wins at depth 1
    probs[0, node_idx[node_b]] = 0.1
    probs[0, node_idx[node_b1]] = 0.5
    probs[0, node_idx[node_b2]] = 0.4

    class MockModule:
        name_to_index = {"genome1": 0}
        classification_tree = root
        probabilities = probs

    df = BarbetLightningModule.generate_top_predictions(MockModule(), barbet2, num_predictions=1)

    ranks_emitted = df["rank"].to_list()
    # Both depth-1 (phylum) and depth-2 (class) rows must be present
    assert "phylum" in ranks_emitted, f"Expected 'phylum' in ranks, got: {ranks_emitted}"
    assert "class" in ranks_emitted, (
        f"Expected 'class' in ranks (single-child regression), got: {ranks_emitted}"
    )
    # Depth-1 top prediction should be 'A'
    phylum_row = df.filter(pl.col("rank") == "phylum")
    assert phylum_row["taxon"][0] == "A"
    # Depth-2 prediction should be 'A1' (the single child of A), inheriting A's probability
    class_row = df.filter(pl.col("rank") == "class")
    assert class_row["taxon"][0] == "A1"
    assert class_row["probability"][0] == pytest.approx(0.9, abs=1e-4)

# ── Global predictions tests ──────────────────────────────────────────────────

def _build_global_mock_tree():
    """Build a two-level mock tree shared by global-predictions tests.

    Tree structure:
        root
        ├── A  (depth 1 / phylum)
        │   ├── A1 (depth 2 / class)
        │   └── A2 (depth 2 / class)
        └── B  (depth 1 / phylum)
            ├── B1 (depth 2 / class)
            └── B2 (depth 2 / class)
    """
    root    = SoftmaxNode("root")
    node_a  = SoftmaxNode("A",  parent=root)
    node_a1 = SoftmaxNode("A1", parent=node_a)
    node_a2 = SoftmaxNode("A2", parent=node_a)
    node_b  = SoftmaxNode("B",  parent=root)
    node_b1 = SoftmaxNode("B1", parent=node_b)
    node_b2 = SoftmaxNode("B2", parent=node_b)
    root.set_indexes()
    return root, node_a, node_a1, node_a2, node_b, node_b1, node_b2


def test_generate_top_predictions_global_mode(tmp_path):
    """Global top-N at each rank spans the entire taxonomy, not just the
    greedy parent's children.

    With phylum A winning and joint-probability order A1 > B1 > A2 > B2,
    top-2 classes globally must return A1 and B1 (from different phyla),
    whereas conditional mode would return only A1 and A2.
    """
    from barbet.modules import BarbetLightningModule

    root, node_a, node_a1, node_a2, node_b, node_b1, node_b2 = _build_global_mock_tree()

    class MockBarbet3:
        def node_to_str(self, node):
            return node.name

    barbet3 = MockBarbet3()

    non_root = [n for n in root.node_list_softmax if not n.is_root]
    node_idx = {n: i for i, n in enumerate(non_root)}

    import torch as _torch
    # Joint probabilities: A=0.70, B=0.30, A1=0.60, A2=0.10, B1=0.25, B2=0.05
    probs = _torch.zeros((1, len(non_root)))
    probs[0, node_idx[node_a]]  = 0.70
    probs[0, node_idx[node_b]]  = 0.30
    probs[0, node_idx[node_a1]] = 0.60
    probs[0, node_idx[node_a2]] = 0.10
    probs[0, node_idx[node_b1]] = 0.25
    probs[0, node_idx[node_b2]] = 0.05

    class MockModule3:
        name_to_index       = {"genome1": 0}
        classification_tree = root
        probabilities       = probs

    # ── global mode, top-2 ──
    df = BarbetLightningModule.generate_top_predictions(
        MockModule3(), barbet3, num_predictions=2, global_predictions=True
    )

    assert list(df.columns) == [
        "name", "rank", "taxon", "joint_probability",
        "local_probability", "prediction_number", "in_predicted_lineage",
    ]

    phylum_df = df.filter(pl.col("rank") == "phylum")
    class_df  = df.filter(pl.col("rank") == "class")

    # Phylum top-2: A (0.70) then B (0.30)
    assert phylum_df["taxon"].to_list() == ["A", "B"]
    assert phylum_df["prediction_number"].to_list() == [1, 2]

    # Class top-2 globally: A1 (0.60) then B1 (0.25) — spans BOTH phyla
    assert class_df["taxon"].to_list() == ["A1", "B1"]
    assert class_df["prediction_number"].to_list() == [1, 2]

    # in_predicted_lineage: greedy path is A → A1
    phylum_lin = dict(zip(phylum_df["taxon"].to_list(), phylum_df["in_predicted_lineage"].to_list()))
    class_lin  = dict(zip(class_df["taxon"].to_list(),  class_df["in_predicted_lineage"].to_list()))
    assert phylum_lin["A"]  is True
    assert phylum_lin["B"]  is False
    assert class_lin["A1"]  is True
    assert class_lin["B1"]  is False   # parent B is not in_predicted_lineage

    # local_probability = joint / parent_joint
    a1_local = class_df.filter(pl.col("taxon") == "A1")["local_probability"][0]
    assert a1_local == pytest.approx(0.60 / 0.70, abs=1e-4)
    b1_local = class_df.filter(pl.col("taxon") == "B1")["local_probability"][0]
    assert b1_local == pytest.approx(0.25 / 0.30, abs=1e-4)

    # ── conditional mode: top-2 classes must be restricted to A's children ──
    df_cond = BarbetLightningModule.generate_top_predictions(
        MockModule3(), barbet3, num_predictions=2, global_predictions=False
    )
    class_cond = df_cond.filter(pl.col("rank") == "class")
    assert set(class_cond["taxon"].to_list()) == {"A1", "A2"}


def test_generate_top_predictions_global_mode_single_child(tmp_path):
    """In global mode, single-child nodes on the greedy path are emitted with
    local_probability=1.0, joint_probability inherited from their parent, and
    in_predicted_lineage=True.

    Tree:
        root
        ├── A        (depth 1, has 1 child — child excluded from softmax)
        │   └── A_only  (depth 2, single child, NOT in node_list_softmax)
        └── B        (depth 1)
            ├── B1   (depth 2)
            └── B2   (depth 2)
    """
    from barbet.modules import BarbetLightningModule

    root        = SoftmaxNode("root")
    node_a      = SoftmaxNode("A",      parent=root)
    node_a_only = SoftmaxNode("A_only", parent=node_a)   # single child → not in softmax
    node_b      = SoftmaxNode("B",      parent=root)
    node_b1     = SoftmaxNode("B1",     parent=node_b)
    node_b2     = SoftmaxNode("B2",     parent=node_b)
    root.set_indexes()

    softmax_set = set(root.node_list_softmax)
    assert node_a_only not in softmax_set, "A_only must be excluded (single child)"

    class MockBarbet4:
        def node_to_str(self, node):
            return node.name

    barbet4 = MockBarbet4()

    non_root = [n for n in root.node_list_softmax if not n.is_root]
    node_idx = {n: i for i, n in enumerate(non_root)}

    import torch as _torch
    probs = _torch.zeros((1, len(non_root)))
    probs[0, node_idx[node_a]]  = 0.80
    probs[0, node_idx[node_b]]  = 0.20
    probs[0, node_idx[node_b1]] = 0.16
    probs[0, node_idx[node_b2]] = 0.04

    class MockModule4:
        name_to_index       = {"genome1": 0}
        classification_tree = root
        probabilities       = probs

    df = BarbetLightningModule.generate_top_predictions(
        MockModule4(), barbet4, num_predictions=3, global_predictions=True
    )

    class_df = df.filter(pl.col("rank") == "class")

    # A_only must appear (on greedy path) even though it is not in the softmax
    a_only_rows = class_df.filter(pl.col("taxon") == "A_only")
    assert len(a_only_rows) == 1
    assert a_only_rows["local_probability"][0] == pytest.approx(1.0, abs=1e-6)
    assert a_only_rows["joint_probability"][0] == pytest.approx(0.80, abs=1e-4)
    assert a_only_rows["in_predicted_lineage"][0] is True
    assert a_only_rows["prediction_number"][0] == 1


def test_generate_top_predictions_global_mode_ranks_only_children(tmp_path):
    """In global mode, only children (absent from node_list_softmax) are ranked with
    the probability of their parent, even when they are not on the greedy path.

    Tree:
        root
        ├── A        (depth 1)
        │   ├── A1   (depth 2)
        │   └── A2   (depth 2)
        └── B        (depth 1, has 1 child)
            └── B_only  (depth 2, single child, NOT in node_list_softmax)

    With A=0.6 (A1=0.35, A2=0.25) and B=0.4 the greedy path is A → A1, but globally
    B_only (0.4) is the most probable class.
    """
    root        = SoftmaxNode("root")
    node_a      = SoftmaxNode("A",      parent=root)
    node_a1     = SoftmaxNode("A1",     parent=node_a)
    node_a2     = SoftmaxNode("A2",     parent=node_a)
    node_b      = SoftmaxNode("B",      parent=root)
    node_b_only = SoftmaxNode("B_only", parent=node_b)
    root.set_indexes()
    assert node_b_only not in set(root.node_list_softmax)

    class MockBarbet5:
        def node_to_str(self, node):
            return node.name

    non_root = [n for n in root.node_list_softmax if not n.is_root]
    node_idx = {n: i for i, n in enumerate(non_root)}
    probs = torch.zeros((1, len(non_root)))
    probs[0, node_idx[node_a]]  = 0.60
    probs[0, node_idx[node_b]]  = 0.40
    probs[0, node_idx[node_a1]] = 0.35
    probs[0, node_idx[node_a2]] = 0.25

    class MockModule5:
        name_to_index       = {"genome1": 0}
        classification_tree = root
        probabilities       = probs

    df = BarbetLightningModule.generate_top_predictions(
        MockModule5(), MockBarbet5(), num_predictions=3, global_predictions=True
    )
    class_df = df.filter(pl.col("rank") == "class")

    assert class_df["taxon"].to_list() == ["B_only", "A1", "A2"]
    assert class_df["prediction_number"].to_list() == [1, 2, 3]
    assert class_df["joint_probability"].to_list() == pytest.approx([0.40, 0.35, 0.25], abs=1e-6)
    assert class_df["local_probability"].to_list() == pytest.approx([1.0, 0.35 / 0.60, 0.25 / 0.60], abs=1e-6)
    assert class_df["in_predicted_lineage"].to_list() == [False, True, False]


def test_predict_context_vector_file(tmp_path):
    barbet = Barbet()
    barbet.load_checkpoint = lambda *args, **kwargs: MockCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()

    output_dir = tmp_path / "output"
    input_file = tmp_path / "0.fa.gz"
    input_file.write_bytes((TEST_DATA_DIR / "MAG-GUT41.fa.gz").read_bytes())

    out_ctx_tsv = output_dir / "context_vectors.tsv"
    results = barbet.predict(
        input=[input_file],
        output_dir=output_dir,
        context_vector_file=out_ctx_tsv,
    )

    assert out_ctx_tsv.exists()
    lines = out_ctx_tsv.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0] == "name\tcontext_vector"
    assert len(lines) > 1
    parts = lines[1].split("\t")
    assert len(parts) == 2
    vec_vals = [float(x) for x in parts[1].split(",")]
    assert len(vec_vals) == 3072
    assert all(v == 0.5 for v in vec_vals)


def test_predict_context_vector_file_gz(tmp_path):
    import gzip

    barbet = Barbet()
    barbet.load_checkpoint = lambda *args, **kwargs: MockCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()

    output_dir = tmp_path / "output"
    input_file = tmp_path / "0.fa.gz"
    input_file.write_bytes((TEST_DATA_DIR / "MAG-GUT41.fa.gz").read_bytes())

    out_ctx_tsv_gz = output_dir / "context_vectors.tsv.gz"
    results = barbet.predict(
        input=[input_file],
        output_dir=output_dir,
        context_vector_file=out_ctx_tsv_gz,
    )

    assert out_ctx_tsv_gz.exists()
    with gzip.open(out_ctx_tsv_gz, "rt", encoding="utf-8") as f:
        content = f.read()
    lines = content.strip().split("\n")
    assert lines[0] == "name\tcontext_vector"
    assert len(lines) > 1
    parts = lines[1].split("\t")
    assert len(parts) == 2
    vec_vals = [float(x) for x in parts[1].split(",")]
    assert len(vec_vals) == 3072
    assert all(v == 0.5 for v in vec_vals)


def test_barbet_lightning_module_context_vector_accumulation(tmp_path):
    import gzip
    from hierarchicalsoftmax import SoftmaxNode
    from barbet.models import BarbetModel

    root = SoftmaxNode('root')
    curr = root
    for r in range(6):
        curr = SoftmaxNode(f'node_{r}', parent=curr)
    root.set_indexes()

    model = BarbetModel(classification_tree=root, features=16, intermediate_layers=0)
    lightning_module = BarbetLightningModule(model=model, loss_function=None, max_learning_rate=0.01, metrics=[])
    lightning_module.hparams.classification_tree = root

    class MockBarbet:
        def node_to_str(self, node):
            return str(node)

    names = ["genome_1", "genome_1", "genome_2"]
    lightning_module.setup_prediction(MockBarbet(), names=names, save_context_vectors=True)

    x1 = torch.randn(2, 32, 16)
    _ = model(x1)
    res1 = torch.zeros((2, root.layer_size))
    lightning_module.on_predict_batch_end(res1, x1, 0)

    x2 = torch.randn(1, 32, 16)
    _ = model(x2)
    res2 = torch.zeros((1, root.layer_size))
    lightning_module.on_predict_batch_end(res2, x2, 1)

    lightning_module.on_predict_epoch_end()

    tsv_path = tmp_path / "out_ctx.tsv.gz"
    lightning_module.save_context_vectors_tsv(tsv_path)

    with gzip.open(tsv_path, "rt", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    assert lines[0] == "name\tcontext_vector"
    assert len(lines) == 3
    g1_parts = lines[1].split("\t")
    assert g1_parts[0] == "genome_1"
    g1_vec = [float(v) for v in g1_parts[1].split(",")]
    assert len(g1_vec) == 16



def _run_epoch_end(**setup_kwargs):
    """Run BarbetLightningModule.on_predict_epoch_end on fixed logits for a tree
    containing both multi-child and single-child nodes."""
    from barbet.models import BarbetModel

    root = SoftmaxNode("root")
    for p in range(2):
        phylum = SoftmaxNode(f"p{p}", parent=root)
        parent = phylum
        for rank in ["c", "o", "f", "g"]:
            parent = SoftmaxNode(f"{parent.name}_{rank}", parent=parent)  # single child
        for s in range(3):
            SoftmaxNode(f"{parent.name}_s{s}", parent=parent)
    root.set_indexes()

    model = BarbetModel(classification_tree=root, features=16, intermediate_layers=0)
    module = BarbetLightningModule(model=model, loss_function=None, max_learning_rate=0.01, metrics=[])
    module.hparams.classification_tree = root

    class MockBarbet:
        def node_to_str(self, node):
            return node.name

    names = [f"genome_{i}" for i in range(4)]
    module.setup_prediction(MockBarbet(), names=names, **setup_kwargs)
    module.on_predict_batch_end(torch.randn((4, root.layer_size), generator=torch.Generator().manual_seed(0)), None, 0)
    module.on_predict_epoch_end()
    category_names = [n.name for n in root.node_list_softmax if not n.is_root]
    return module, category_names


def test_save_probabilities_includes_node_columns():
    """Regression test: save_probabilities (predict_memmap --probabilities) must include a
    probability column for every node in the taxonomy."""
    module, category_names = _run_epoch_end(save_probabilities=True)
    columns = module.results_df.columns
    assert columns[:13] == ["name"] + [f"{r}_{c}" for r in RANKS for c in ("prediction", "probability")]
    assert columns[13:] == category_names


def test_retain_probabilities_does_not_change_results():
    """retain_probabilities (used by --output-predictions) keeps the probability matrix
    without changing barbet-predictions.csv."""
    baseline, _ = _run_epoch_end()
    retained, category_names = _run_epoch_end(retain_probabilities=True)

    assert baseline.probabilities is None
    assert retained.probabilities is not None
    assert tuple(retained.probabilities.shape) == (4, len(category_names))
    assert retained.results_df.equals(baseline.results_df)


@pytest.mark.parametrize(
    "path, expected",
    [
        (True, "out/default.tsv"),
        ("true", "out/default.tsv"),
        ("1", "out/default.tsv"),
        ("existing_dir", "existing_dir/default.tsv"),
        ("preds.tsv", "out/preds.tsv"),
        ("sub/preds.tsv", "sub/preds.tsv"),
    ],
)
def test_resolve_output_path(path, expected, tmp_path, monkeypatch):
    from barbet.apps import resolve_output_path

    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing_dir").mkdir()

    resolved = resolve_output_path(path, Path("out"), "default.tsv")

    assert resolved == Path(expected)
    assert resolved.parent.is_dir()


def test_resolve_output_path_absolute(tmp_path):
    from barbet.apps import resolve_output_path

    path = tmp_path / "nested" / "preds.tsv"
    assert resolve_output_path(path, tmp_path / "out", "default.tsv") == path
    assert path.parent.is_dir()


def test_model_forward_does_not_store_context_vector():
    """Regression test: BarbetModel.forward must not keep the context vector on the model. The whole
    model is pickled into checkpoints, and deepcopy fails for a stored non-leaf tensor after a
    training step."""
    import copy
    from barbet.models import BarbetModel

    root = SoftmaxNode("root")
    for i in range(3):
        SoftmaxNode(f"n{i}", parent=root)
    root.set_indexes()

    model = BarbetModel(classification_tree=root, features=16, intermediate_layers=0)
    model.train()
    model(torch.randn(2, 4, 8)).sum().backward()

    assert not any(torch.is_tensor(value) for value in vars(model).values())
    copy.deepcopy(model)


def test_context_vector_hook_captures_classifier_input_and_is_removed():
    from barbet.models import BarbetModel

    # on_predict_epoch_end requires a lineage for every rank
    root = SoftmaxNode("root")
    for p in range(2):
        parent = SoftmaxNode(f"p{p}", parent=root)
        for rank in ["c", "o", "f", "g", "s"]:
            parent = SoftmaxNode(f"{parent.name}_{rank}", parent=parent)
    root.set_indexes()

    model = BarbetModel(classification_tree=root, features=16, intermediate_layers=0)
    module = BarbetLightningModule(model=model, loss_function=None, max_learning_rate=0.01, metrics=[])
    module.hparams.classification_tree = root

    class MockBarbet:
        def node_to_str(self, node):
            return node.name

    def context_vector(x):
        # Same calculation as BarbetModel.forward up to the classification layer
        x = model.sequential(x)
        attention_weights = torch.softmax(model.attention_layer(x), dim=1)
        return torch.sum(attention_weights * x, dim=1)

    x = torch.randn(3, 4, 8)
    model.eval()
    with torch.no_grad():
        model(x)  # initialise the lazy layers
        module.setup_prediction(MockBarbet(), names=["g1", "g1", "g2"], save_context_vectors=True)
        results = model(x)
        module.on_predict_batch_end(results, x, 0)
        module.on_predict_epoch_end()
        expected = context_vector(x)

    assert torch.allclose(module.context_vectors[0], expected[:2].mean(dim=0), atol=1e-6)
    assert torch.allclose(module.context_vectors[1], expected[2], atol=1e-6)
    assert len(model.classifier._forward_pre_hooks) == 0


def test_predict_writes_results_before_optional_outputs(tmp_path):
    """barbet-predictions.csv must be written even if writing an optional output fails."""

    class FailingCheckpoint(MockCheckpoint):
        def save_context_vectors_tsv(self, output_path):
            raise RuntimeError("cannot write context vectors")

    barbet = Barbet()
    barbet.load_checkpoint = lambda *args, **kwargs: FailingCheckpoint()
    barbet.prediction_trainer = lambda *args, **kwargs: MockPredictionTrainer()

    output_dir = tmp_path / "output"
    input_file = tmp_path / "0.fa.gz"
    input_file.write_bytes((TEST_DATA_DIR / "MAG-GUT41.fa.gz").read_bytes())

    with pytest.raises(RuntimeError, match="cannot write context vectors"):
        barbet.predict(input=[input_file], output_dir=output_dir, context_vector_file="ctx.tsv")

    assert (output_dir / "barbet-predictions.csv").exists()
