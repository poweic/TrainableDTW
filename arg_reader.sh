# ===========================
# ===== Input Arguments =====
# ===========================
IV_OR_OOV=$1
if [ "$IV_OR_OOV" == "IV" ]; then
  N_QUERY=275
  CORPUS=SI_word
elif [ "$IV_OR_OOV" == "OOV" ]; then
  N_QUERY=110
  CORPUS=OOV_g2p
else
  printf "Choose either \33[32m IV \33[0m or \33[32m OOV \33[0m \n"
  exit -1;
fi

EXP_SET=$2
if [ "$EXP_SET" == "" ]; then
  printf "Choose a non-empty experiment set suffix. \33[31m (NOT \"\") \33[0m \n"
  exit -1;
fi

FEAT_TYPE=$3
if [ "$FEAT_TYPE" == "" ]; then
  printf "Choose either \33[32m mfcc (39) \33[0m or \33[32m posterior (76) \33[0m. \n"
  exit -1;
fi

ark="/media/Data1/hypothesis/SI_word.kaldi/mfcc/[A457][ADAD].39.ark"

shift; shift; shift;
opts=("$@")

# ============================
# ===== Output Arguments =====
# ============================

QUERY_LIST=/home/boton/Dropbox/DSP/RandomWalk/source/query/${N_QUERY}.query
hypo_dir=/share/hypothesis/${CORPUS}.kaldi
mfcc_dir=${hypo_dir}/mfcc/
posterior_dir=${hypo_dir}/posterior/

if [ "$FEAT_TYPE" == "mfcc" ]; then
  feature_dir=$mfcc_dir
  feature_dim=39
elif [ "$FEAT_TYPE" == "posterior" ]; then
  feature_dir=$posterior_dir
  feature_dim=76
else
  printf "\33[31m Illegal feature type \33[0m\n"
  exit -1;
fi
#MFCC_DIR=/share/hypothesis/${CORPUS}_gp/
PRE_DIR=/share/preparsed_files
DIR=$PRE_DIR/${CORPUS}_${EXP_SET}
MULSIM_DIR=${DIR}/mul-sim
LIST_DIR=${DIR}/seglist

if [ "$EXP_SET" == "dtw" ]; then
  dtw_options="--dtw-type=fixdtw"
else
  dtw_options="--dtw-type=cdtw"
fi

printf "Vocabulary \33[32m ${IV_OR_OOV} \33[0m \n"
printf "# of queries = \33[32m ${N_QUERY} \33[0m \n"
printf "Experiments sets for Dynamic Time Warping: \33[32m${EXP_SET}\33[0m\n"
printf "Output Directory for mul-sim: \33[32m${MULSIM_DIR}\33[0m\n"
printf "Additional arguments for calc-acoustic-similarity: \33[32m \" ${opts} \" \33[0m \n"
printf "Using option \"\33[34m$dtw_options\33[0m\"\n"
