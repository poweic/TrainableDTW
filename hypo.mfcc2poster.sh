#N_QUERY=275
#corpus=SI_word
N_QUERY=110
corpus=OOV_g2p

QUERY_LIST=/home/boton/Dropbox/DSP/RandomWalk/source/query/${N_QUERY}.query
QUERIES=($(cat $QUERY_LIST))

hypo_root=/share/hypothesis/
HYPO_DIR=$hypo_root/$corpus/
SegListDir=/share/preparsed_files/$corpus/seglist/
mfcc_dir=$hypo_root/$corpus.kaldi/mfcc
posterior_dir=$hypo_root/$corpus.kaldi/posterior

mkdir -p $mfcc_dir
mkdir -p $posterior_dir

BIN=./htk-to-kaldi
converter=/share/mlp_posterior/calc-posterior-gram.sh

for q in ${QUERIES[@]}; do
  echo $q
  $BIN -d ${HYPO_DIR}/$q --list=${SegListDir}/$q.lst --extension=.mfc > ${mfcc_dir}/$q.39.ark 
  ${converter} ${mfcc_dir}/$q.39.ark ${posterior_dir}/$q.76.ark
done
