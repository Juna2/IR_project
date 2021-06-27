result=${PWD##*/}
echo ${result}_tmp_201123

cd ../
rsync -av --progress ./$result/ ./${result}_tmp_201123 --exclude result/* --exclude result_important/* --exclude data/* --exclude history/*
