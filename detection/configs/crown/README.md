# CROWN detection integration

The Faster R-CNN and Mask R-CNN templates are in
`../faster_rcnn/faster_rcnn_crown_adapter_large_fpn_1x_coco.py` and
`../mask_rcnn/mask_rcnn_crown_adapter_large_fpn_1x_coco.py`.

Both use the official CROWN ViT-L/16 definition and load `CROWN.pth` with
`strict=True`. The original 224-pixel positional embeddings are interpolated
by the official model for detection inputs. The class token stays in every
transformer block, while patch tokens interact with ViT-Adapter's spatial
prior. The FPN receives strides 4, 8, 16, and 32.

The input pipeline converts images to RGB and uses ImageNet mean and standard
deviation, as in CROWN's `data/transforms.py` training and evaluation
transforms. CROWN's README also has a minimal feature-extraction example with
only `ToTensor()`; that example does not use the official evaluation
normalization. Detection resize and crop settings are task-specific; padding
to a multiple of 32 is required by the adapter.

To run on a new dataset, inherit one of the templates and override its
`data_root`, annotations, image prefixes, classes, and `num_classes` in the
ROI head. Keep the RGB normalization in train, validation, and test pipelines.
The pretrained checkpoint and official source paths can be changed through
`pretrained` and `official_root` in the config.

From the repository root, a template can be launched with:

```bash
source env.sh
cd detection
python train.py configs/faster_rcnn/faster_rcnn_crown_adapter_large_fpn_1x_coco.py
```

## Full dataset queue

The runner discovers each COCO split, writes configs and progress CSVs, and
schedules the smallest training sets first. It reads directly from the native
NFS mount on this machine; no sshfs mount or local dataset copy is required.
The default physical GPUs are 4, 5, 6, and 7, one experiment per GPU. Override
these with `CROWN_GPUS=0,1,2,3` (or another comma-separated list).

```bash
scripts/crown_pipeline.sh prepare  # validate splits and generate configs
scripts/crown_pipeline.sh start    # train, select on validation, test, archive
scripts/crown_pipeline.sh status   # show progress and output paths
scripts/crown_pipeline.sh stop     # stop all queue processes
```

The launcher resolves Python from the micromamba `torch29` environment using
`MAMBA_ROOT_PREFIX` or `~/micromamba`, with `micromamba run` as a fallback.
Set `CROWN_PYTHON` to use an explicit Python executable.

Default paths on this machine:

| Purpose | Default | Override |
| --- | --- | --- |
| COCO datasets | `/jhcnas6/Public` | `CROWN_PUBLIC_ROOT` |
| Pretrained weights | `/jhcnas6/Private/temp/X_Ckpts/crown_ckpts/CROWN.pth` | `CROWN_PRETRAINED_CKPT` |
| Official CROWN source | `~/0_Official/CROWN` | `CROWN_OFFICIAL_ROOT` |
| NAS training, evaluation, logs and checkpoints | `/jhcnas6/Cytology/smartcyto_baseline/crown` | `CROWN_ARCHIVE_ROOT` |
| Optional dataset staging | `<repo>/work_dirs/crown_runs/data` | `CROWN_LOCAL_ROOT` |
| Generated configs and progress CSVs | `<repo>/work_dirs/crown_pipeline` | `CROWN_STATE_ROOT` |

`CROWN_MOUNTPOINT` changes the default NAS root (`/jhcnas6`). Individual path
variables take precedence. The templates also honor `CROWN_PRETRAINED_CKPT`
and `CROWN_OFFICIAL_ROOT` when launched manually.

`prepare` only needs readable datasets. Before `start` launches experiments,
the runner checks CUDA device availability, official source, and NAS
output permissions (plus rsync when input staging is enabled). The current archive parent must be accessible and
writable by your account. If it is restricted, fix its permissions or set
`CROWN_ARCHIVE_ROOT` to an accessible destination before starting.

The runner reads JSON files, images, and pretrained weights directly from
NAS by default. It loads pretrained tensors into CPU memory without memory
mapping. Set `CROWN_STAGE_INPUTS=1` before `prepare` and `start` to retain the
old resumable local staging mode for slower or unreliable storage. Use the
same setting throughout a run. That mode copies the pretrained checkpoint to
`<archive>/_pretrained/CROWN.pth`, verifies its ZIP CRC, and stages only the images
referenced in each dataset's COCO JSON. A separate CPU worker copies data,
and removes staged inputs after successful archiving. The NAS source files
are never removed.

Progress is recorded in `status.csv` and `results.csv` under the state root.
The latter contains one row per metric with its mean and 95% bootstrap
interval. Training and evaluation write directly into
`/jhcnas6/Cytology/smartcyto_baseline/crown/{det,seg}/<dataset>/` (or the
corresponding `CROWN_ARCHIVE_ROOT` destination). All epoch, latest, and best
checkpoints are saved there from the beginning; no training checkpoint is
written to local disk. Logs, evaluation results and completion markers also
live there. A separate CPU worker finalizes the saved config without copying
the directory onto itself. After evaluation and finalization succeed, only
the selected best checkpoint is retained; interrupted runs keep their latest
checkpoint for resumption. TXL-PBC's best checkpoint remains on NAS permanently
and CBC uses it for external testing. `CROWN_LOCAL_ROOT` only controls optional
dataset staging, not training checkpoint storage.

Faster R-CNN selects `bbox_mAP` on validation. Mask R-CNN selects AJI on
validation. Both use the 1x schedule, validate each epoch, and stop after five
evaluation rounds without improvement. The final test uses only the saved best
checkpoint, and computes 1000 image-level bootstrap resamples. Detection
reports mAP, AP30, AP50, and mAR; segmentation reports AJI, Dice, mAP, and
AP50. CBC is an external detection test using TXL-PBC's checkpoint. Its class
order is mapped to the TXL-PBC head (WBC, RBC, Platelets).

The runner probes the NAS dataset directory every 15 seconds with a time
limit, independent of filesystem type. After three consecutive failures,
active process groups stop and their rows become `paused_mount`. Restore NAS
access and run `start` again; training resumes from `latest.pth` and completed
stages are skipped. A failed task can also be retried with `start` after
fixing its underlying issue. Failed finalization retains NAS checkpoints for retry.
