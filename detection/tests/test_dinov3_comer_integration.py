import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKBONE = ROOT / "detection" / "mmdet_custom" / "models" / "backbones" / "dinov3_comer.py"
INIT = ROOT / "detection" / "mmdet_custom" / "models" / "backbones" / "__init__.py"
CONFIG = ROOT / "my-configs" / "cps_maskrcnn_dinov3_vit_comer_base_fpn_100e.py"
SLURM = ROOT / "scripts" / "slurm_cps_vit_comer_dinov3.sh"


def _source(path):
    return path.read_text()


def _tree(path):
    return ast.parse(_source(path))


def test_dinov3_comer_backbone_is_registered_and_uses_official_dinov3():
    source = _source(BACKBONE)
    tree = _tree(BACKBONE)

    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "ViTCoMerDINOv3" in class_names
    assert "DINOv3CoMer" in source
    assert "OfficialDINOv3Backbone.from_checkpoint" in source
    assert "normalize_interaction_ranges" in source
    assert "@BACKBONES.register_module()" in source

    init_source = _source(INIT)
    assert "ViTCoMerDINOv3" in init_source
    assert "DINOv3CoMer" in init_source


def test_dinov3_comer_preserves_comer_two_branch_interaction_path():
    source = _source(BACKBONE)

    assert "from .comer_modules import CNN, CTIBlock, deform_inputs" in source
    assert "self.spm = CNN" in source
    assert "class DINOv3CTIBlock(CTIBlock)" in source
    assert "self.cti_tov" in source
    assert "self.cti_toc" in source
    assert "torch.cat((prefix_tokens, x), dim=1)" in source
    assert "block(hidden_states, position_embeddings=position_embeddings)" in source
    assert "hidden_states = torch.cat((prefix_tokens, patch_tokens), dim=1)" in source
    assert "outs = []" not in source
    assert "x1, x2, x3, x4 = outs" not in source
    assert "x3 = patch_tokens.transpose" in source


def test_dinov3_comer_config_and_slurm_entrypoint_exist():
    config_source = _source(CONFIG)
    slurm_source = _source(SLURM)

    assert "type='ViTCoMerDINOv3'" in config_source
    assert "adapter_mode" not in config_source
    assert "dinov3-vitb16-pretrain-lvd1689m" in config_source
    assert "work_dirs/segmentation/cps_maskrcnn_dinov3_vit_comer_base_fpn_100e" in config_source

    assert "cps_maskrcnn_dinov3_vit_comer_base_fpn_100e.py" in slurm_source
    assert "micromamba" in slurm_source
    assert "source \"$ROOT_DIR/env.sh\"" in slurm_source
    assert "detection/dist_train.sh" in slurm_source
    assert "run_experiment_eval.py" in slurm_source


def test_dinov3_comer_freezes_unused_dinov3_params_to_avoid_ddp_reduction_errors():
    source = _source(BACKBONE)

    assert 'getattr(self.backbone.model.embeddings, "mask_token", None)' in source
    assert 'getattr(self.backbone.model.norm, "weight", None)' in source
    assert 'getattr(self.backbone.model.norm, "bias", None)' in source
    assert "requires_grad_(False)" in source
