package com.github.tjake.rbm;

import java.util.Iterator;
import java.util.List;

public class SimpleRBMTrainer
{
    public float momentum;
    final float l2;
    final Float targetSparsity;
    public float learningRate;
    private LayerFactory layerFactory;

    Layer[] gw;
    Layer gv;
    Layer gh;

    public SimpleRBMTrainer(float momentum, float l2, Float targetSparsity, Float learningRate, LayerFactory layerFactory)
    {
        this.momentum = momentum;
        this.l2 = l2;
        this.targetSparsity = targetSparsity;
        this.learningRate = learningRate;
        this.layerFactory = layerFactory;
    }

    // 실제로 러닝이 일어나는 부분
    public double learn(final SimpleRBM rbm, List<Layer> inputBatch, boolean reverse)
    {
        int batchsize = inputBatch.size();

        if (gw == null || gw.length != rbm.biasHidden.size() || gw[0].size() != rbm.biasVisible.size())
        {
            gw = new Layer[rbm.biasHidden.size()];
            for(int i=0; i<gw.length; i++)
                gw[i] = layerFactory.create(rbm.biasVisible.size());

            gv = layerFactory.create(rbm.biasVisible.size());
            gh = layerFactory.create(rbm.biasHidden.size());
        }
        else
        {
            for(int i=0; i<gw.length; i++)
                gw[i].clear();

            gv.clear();
            gh.clear();
        }
        
        // Contrastive Divergance:: 실제 러닝
        for (Layer input : inputBatch) //30개의 인풋 배치에 속하는 각각의 인풋에 대해
        {
            try {
                Iterator<Tuple> it = reverse ? rbm.reverseIterator(input) : rbm.iterator(input);
                //마르코프 체인부분인것 같음
                Tuple t1 = it.next();    //UP
                Tuple t2 = it.next();    //Down

                for (int i=0; i< gw.length; i++)
                    for (int j=0; j<gw[i].size(); j++)
                        gw[i].add(j, (t1.hidden.get(i) * t1.visible.get(j)) - (t2.hidden.get(i) * t2.visible.get(j)));

                for (int i = 0; i < gv.size(); i++)
                    gv.add(i, t1.visible.get(i) - t2.visible.get(i));

                for (int i = 0; i < gh.size(); i++)
                    gh.add(i,  targetSparsity == null ? t1.hidden.get(i) - t2.hidden.get(i) : targetSparsity - t1.hidden.get(i));

            } catch (Throwable t) {
                t.printStackTrace();
            }
        }


        // Average
        for (int i = 0; i < gw.length; i++)
        {
            for (int j = 0; j < gw[i].size(); j++)
            {
                gw[i].div(j, batchsize);

                gw[i].mult(j, 1 - momentum);
                gw[i].add(j,  momentum * (gw[i].get(j) - l2*rbm.weights[i].get(j)));
                //웨이트 수정
                rbm.weights[i].add(j, learningRate * gw[i].get(j));
            }
        }

        double error = 0.0;

        for (int i = 0; i < gv.size(); i++)
        {
            gv.div(i, batchsize);

            error += Math.pow(gv.get(i), 2);

            gv.mult(i, 1 - momentum);
            gv.add(i, momentum * (gv.get(i) * rbm.biasVisible.get(i)));
            //비지블 바이어스도 수정함
            rbm.biasVisible.add(i, learningRate * gv.get(i));
        }

        error = Math.sqrt(error/gv.size());

        if (targetSparsity != null)
        {
            for (int i=0; i<gh.size(); i++)
            {
                gh.div(i,batchsize);
                gh.set(i, targetSparsity - gh.get(i));
            }
        }
        else
        {
            for (int i = 0; i < gh.size(); i++)
            {
                gh.div(i, batchsize);

                gh.mult(i, 1 - momentum);
                gh.add(i, momentum * (gh.get(i) * rbm.biasHidden.get(i)));

                rbm.biasHidden.add(i, learningRate * gh.get(i));
            }
        }


        return error;
    }
}
