#!/bin/bash
curl https://sdk.cloud.google.com --output /root/googlesdk.installer && source /root/googlesdk.installer --disable-prompts && source /root/google-cloud-sdk/path.bash.inc && mkdir -p /root/coco/val2017 && gsutil -m rsync gs://images.cocodataset.org/val2017 /root/coco/val2017
