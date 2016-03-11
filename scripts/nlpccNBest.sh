make clean
make LSTMWordDetector

scripts=../Release/
corpus=../../acl2016data/nlpcc2014/
Embdata=../../acl2016data/mergTangPenny/

for i in `seq 1 16`;
do
    $scripts/LSTMWordDetector -l -train $corpus/nlpcc.best$i.ctb.train -dev $data/nlpcc.best$i.ctb.dev -test $data/nlpcc.best$i.ctb.test -word $Embdata/weibo.noface.ctb.addUNKWord.50d.vect -option ../options/option.nbest.50d.do25.word >ctb.best$i.50d.do25.LSTM.log 2>&1
done