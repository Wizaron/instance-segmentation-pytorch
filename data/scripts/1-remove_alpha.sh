for f in ../raw/CVPPP2017_LSC_training/training/A1/*_rgb.png;
do
  convert $f -alpha off $f
done
