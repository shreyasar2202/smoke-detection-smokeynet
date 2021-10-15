mkdir /root/pytorch_lightning_data &&
cp /userdata/kerasData/data/new_data/raw_images.tar.gz /root/ &&
tar -zxf /root/raw_images.tar.gz -C /root/ &&
rm -rf /root/raw_images.tar.gz &&
cp /userdata/kerasData/data/new_data/pytorch_lightning_data/drive_clone_numpy_nearest_contour.tar.gz /root/pytorch_lightning_data &&
tar -zxf /root/pytorch_lightning_data/drive_clone_numpy_nearest_contour.tar.gz -C /root/pytorch_lightning_data/ &&
rm -rf /root/pytorch_lightning_data/drive_clone_numpy_nearest_contour.tar.gz
# cp /userdata/kerasData/data/new_data/raw_images_mog.tar.gz /root/ &&
# tar -zxf /root/raw_images_mog.tar.gz -C /root/ &&
# rm -rf /root/raw_images_mog.tar.gz