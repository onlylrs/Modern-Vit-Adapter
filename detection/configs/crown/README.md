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

The one-command runner discovers each requested COCO split and writes local
configs and a progress CSV. It schedules the smallest training sets first and
uses physical GPUs 4, 5, 6, and 7, with one experiment process per GPU.
It detects whether sshfs is rooted at `/jhcnas6` or
`/jhcnas6/Cytology` and adjusts the NAS paths accordingly.

```bash
scripts/crown_pipeline.sh prepare  # validate splits and generate configs
scripts/crown_pipeline.sh start    # train, select on validation, test, archive
scripts/crown_pipeline.sh status   # show progress and output paths
scripts/crown_pipeline.sh stop     # stop all queue processes
```

The launcher uses the existing `torch29` Python directly. Override it with
`CROWN_PYTHON` if the environment moves.

Before scheduling any jobs, the runner copies the 1.2 GB CROWN source
checkpoint to `/homes/rliuar/work/2_Temp/crown_runs/pretrained/CROWN.pth`,
checks its size and ZIP CRC, then points generated configs at that local copy.
This avoids memory-mapped reads from sshfs during GPU startup. An interrupted
copy resumes on the next `start`. Archived configs point back to the NAS
source, and the staged copy is removed after all tasks complete.

Each job first copies its COCO JSON files and only the referenced images to
`/homes/rliuar/work/2_Temp/crown_runs/data/<task>`. The copy uses rsync and
continues from partial files on the next `start`. Training and test then read
the local copy, avoiding intermittent sshfs image read errors. One data copy
runs at a time. The local data copy is removed after that job has been
archived successfully.
Data copying and checkpoint archiving use separate CPU workers, leaving all
four GPU slots available for experiments when datasets are ready.

The progress files are `work_dirs/crown_pipeline/status.csv` and
`work_dirs/crown_pipeline/results.csv`. The latter contains one row per metric
with its mean and 95% bootstrap interval. Each task's logs and checkpoint
resumption files live under `/homes/rliuar/work/2_Temp/crown_runs` and are
synced to the NAS `smartcyto_baseline/crown/{det,seg}` directory. Local
checkpoints are deleted only after `rsync` succeeds. Set `CROWN_LOCAL_ROOT`,
`CROWN_STATE_ROOT`, `CROWN_PUBLIC_ROOT`, or `CROWN_ARCHIVE_ROOT` to override
those paths.
Checkpoint archiving also retries interrupted sshfs writes with resumable
rsync transfers. It does not request NAS owner, group, or permission changes,
and keeps local checkpoints until the full archive succeeds. Only the best
validation checkpoint is archived; intermediate epoch checkpoints are removed
locally after that succeeds.
The TXL-PBC best checkpoint stays in the local temporary directory until
CBC's external test completes, so that test does not load a model across sshfs.

Faster R-CNN selects `bbox_mAP` on validation. Mask R-CNN selects AJI on
validation. Both use the 1x schedule, validate each epoch, and stop after five
evaluation rounds without improvement. The final test uses only the saved best
checkpoint, and computes 1000 image-level bootstrap resamples. Detection
reports mAP, AP30, AP50, and mAR; segmentation reports AJI, Dice, mAP, and
AP50. CBC is an external detection test using TXL-PBC's checkpoint. Its class
order is mapped to the TXL-PBC head (WBC, RBC, Platelets).

The runner checks the sshfs mount every 15 seconds. If it disappears, active
process groups are stopped and their rows become `paused_mount`. Remount the
NAS and run `start` again; training resumes from `latest.pth`, while finished
stages are skipped. A failed task can likewise be retried with `start` after
the underlying issue is fixed.
The health check also probes dataset and archive directories with a time
limit, since an sshfs mount can remain listed while file reads hang. Set
`CROWN_MOUNTPOINT` if the mount path moves.
