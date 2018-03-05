model=${1}
n_workers=4
img_dir=../data/raw/cityscapes/leftImg8bit/val
out_dir=../outputs/cityscapes

model_dir=`dirname $model`
model_dir=`basename ${model_dir}`
model_name=`basename $model .pth`
out_dir=${out_dir}/${model_dir}-${model_name}

mkdir -p ${out_dir}

while read line;
do
  _dir=`echo $line | cut -f1 -d"_"`
  python pred.py --image ${img_dir}/${_dir}/${line}_leftImg8bit.png \
    --model $model --usegpu --output ${out_dir}/${line} \
    --n_workers ${n_workers} --dataset cityscapes
done < ../data/metadata/cityscapes/validation.lst
