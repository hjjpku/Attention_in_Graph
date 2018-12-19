#!/bin/bash

# input arguments
NAME="${1-train}"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
DATA="${2-DD}"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
fold=${3-1}  # which fold as testing data
test_number=${4-0}  # if specified, use the last test_number graphs as test data

# general settings
gm=DGCNN  # model
gpu_or_cpu=gpu
CONV_SIZE="32-32-32-1"
sortpooling_k=0.6  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=0  # final dense layer's input dimension, decided by data
bsize=20  # batch size

# dataset-specific settings
case ${DATA} in
MUTAG)
  num_epochs=300
  ;;
ENZYMES)
  num_epochs=500
  ;;
NCI1)
  num_epochs=200
  ;;
NCI109)
  num_epochs=200
  ;;
DD)
  num_epochs=500
  ;;
PTC)
  num_epochs=200
  ;;
PROTEINS)
  num_epochs=100
  ;;
COLLAB)
  num_epochs=300
  sortpooling_k=0.9
  ;;
IMDBBINARY)
  num_epochs=300
  sortpooling_k=0.9
  ;;
IMDBMULTI)
  num_epochs=500
  sortpooling_k=0.9
  ;;
*)
  num_epochs=500
  ;;
esac

if [ ${fold} == 0 ]; then
  rm result.txt
  echo "Running 10-fold cross validation"
  start=`date +%s`
  for i in $(seq 1 10)
  do
    python main.py \
        -seed 1 \
		-name $NAME \
        -data $DATA \
        -fold $i \
        -num_epochs $num_epochs \
        -latent_dim $CONV_SIZE \
        -sortpooling_k $sortpooling_k \
        -batch_size $bsize \
        -mode $gpu_or_cpu
  done
  stop=`date +%s`
  echo "End of cross-validation"
  echo "The total running time is $[stop - start] seconds."
  echo "The accuracy results for ${DATA} are as follows:"
  cat result.txt
  echo "Average accuracy is"
  cat result.txt | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
else
  python main.py \
      -seed 1 \
      -name $NAME \
      -data $DATA \
      -fold $fold \
      -num_epochs $num_epochs \
      -latent_dim $CONV_SIZE \
      -sortpooling_k $sortpooling_k \
      -batch_size $bsize \
      -mode $gpu_or_cpu \
      -test_number ${test_number}
fi
