model=${1}
n_workers=4
img_dir=../data/raw/CVPPP2017_LSC_training/training/A1
out_dir=../outputs

model_dir=`dirname $model`
model_dir=`basename ${model_dir}`
model_name=`basename $model .pth`
out_dir=${out_dir}/${model_dir}-${model_name}

mkdir -p ${out_dir}

while read line;
do
  python pred.py --image ${img_dir}/${line}_rgb.png --model $model --usegpu --output ${out_dir}/${line} --n_workers ${n_workers}
done < ../data/metadata/validation.lst
