function guard() {
  printf "\33[31mAre you sure you want to do this?\33[0m"
  read -p " (yes/no) " sure
  if [ "$sure" != "yes" ]; then
    exit
  fi
}

guard


PHONES=`cat data/phones.txt | awk '{print $1"\n"}'`

counter=0
for p in $PHONES; do
  #if [ "$p" == "<eps>" ] || [ "$p" == "sil" ] ; then continue; fi


  FOLDER=data/mfcc/$p
  mkdir -p data/train/list/
  LIST=data/train/list/$counter.list
  let counter=counter+1
  rm -f $LIST

  printf "Shuffing phone \33[33m%6s\33[0m and save it into \33[34m$LIST\33[0m\n" $p
  ls $FOLDER | shuf > $LIST

done
