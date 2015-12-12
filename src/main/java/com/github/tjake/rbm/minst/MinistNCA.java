package com.github.tjake.rbm.minst;

import com.github.tjake.rbm.*;

import java.io.*;
import java.util.*;

public class MinistNCA implements Serializable{
    public static MinstDatasetReader dr;
    StackedRBM rbm;
    StackedRBMTrainer trainer;
    LayerFactory layerFactory = new LayerFactory();

    public MinistNCA(File labels, File images) {
        dr = new MinstDatasetReader(labels, images);

        rbm = new StackedRBM();
        trainer = new StackedRBMTrainer(rbm, 0.9f, 0.001f, 0.2f, 0.1f, layerFactory);
        //TODO: decay weights were initialized with small random values sampled from a zero ND with variance 0.01
    }


    public MinistNCA() {
    }

    void learn_epoch (int epoch, int mini_batch_size, boolean addLabels, int stopAt) {

        for (int e = 0; e < epoch; e++) {

            Iterator<MinstItem> iter = dr.iterator();
            // Get random input
            double error=0;
            while(iter.hasNext()) {

                error = minibatch(mini_batch_size, iter, addLabels, stopAt);

            }


            System.err.println("Epoch: " + e +", Error = " + error+", Energy = "+rbm.freeEnergy());
        }
    }

    private double minibatch(int mini_batch_size, Iterator<MinstItem> iter, boolean addLabels, int stopAt) {

        List<Layer> inputBatch = new ArrayList<Layer>();
        List<Layer> labelBatch = addLabels ? new ArrayList<Layer>() : null;
        int size = 0;

        while(iter.hasNext() && size < mini_batch_size){

            MinstItem trainItem = iter.next();
            Layer input = layerFactory.create(trainItem.data.length);

            for (int i = 0; i < trainItem.data.length; i++)
                input.set(i, trainItem.data[i]);

            inputBatch.add(new BinaryLayer(input));
            size++;
        }

        return trainer.learn(inputBatch, labelBatch, stopAt);



    }


    void learn(int iterations, boolean addLabels, int stopAt) {

        for (int p = 0; p < iterations; p++) {

            // Get random input
            List<Layer> inputBatch = new ArrayList<Layer>();
            List<Layer> labelBatch = addLabels ? new ArrayList<Layer>() : null;


            for (int j = 0; j < 100; j++) {
                MinstItem trainItem = dr.getTrainingItem();
                Layer input = layerFactory.create(trainItem.data.length);

                for (int i = 0; i < trainItem.data.length; i++)
                    input.set(i, trainItem.data[i]);

                inputBatch.add(new BinaryLayer(input));

                if (addLabels) {
                    float[] labelInput = new float[10];
                    labelInput[Integer.valueOf(trainItem.label)] = 1.0f;
                    labelBatch.add(layerFactory.create(labelInput));
                }
            }

            double error = trainer.learn(inputBatch, labelBatch, stopAt);

            if (p % 100 == 0)
                System.err.println("Iteration " + p + ", Error = " + error+", Energy = "+rbm.freeEnergy());
        }
    }

    Iterator<Tuple> evaluate(MinstItem test) {

        Layer input = layerFactory.create(test.data.length);

        for (int i = 0; i < test.data.length; i++)
            input.set(i, test.data[i]);

        input = new BinaryLayer(input);

        int stackNum = rbm.getInnerRBMs().size();

        for (int i = 0; i < stackNum; i++) {

            SimpleRBM iRBM = rbm.getInnerRBMs().get(i);

            if (iRBM.biasVisible.size() > input.size()) {
                Layer newInput = new Layer(iRBM.biasVisible.size());

                System.arraycopy(input.get(), 0, newInput.get(), 0, input.size());
                for (int j = input.size(); j < newInput.size(); j++)
                    newInput.set(j, 0.1f);

                input = newInput;
            }

            if (i == (stackNum - 1)) {
                return iRBM.iterator(input);
            }

            input = iRBM.activateHidden(input, null);
        }

        return null;
    }

    Layer feedfoward(MinstItem test) {

        Layer input = layerFactory.create(test.data.length);

        for (int i = 0; i < test.data.length; i++)
            input.set(i, test.data[i]);

        input = new BinaryLayer(input);

        int stackNum = rbm.getInnerRBMs().size();

        for (int i = 0; i < stackNum; i++) {

            SimpleRBM iRBM = rbm.getInnerRBMs().get(i);

            if (iRBM.biasVisible.size() > input.size()) {
                Layer newInput = new Layer(iRBM.biasVisible.size());

                System.arraycopy(input.get(), 0, newInput.get(), 0, input.size());
                for (int j = input.size(); j < newInput.size(); j++)
                    newInput.set(j, 0.1f);

                input = newInput;
            }

            if (i == (stackNum - 1)) {
                return iRBM.activateHidden(input, null);
            }

            input = iRBM.activateHidden(input, null);
        }

        return null;
    }


    public static void pretraining(File labels, File images) {

        //revisited
//        BinaryMinstDBN m = new BinaryMinstDBN(labels,images);
        MinistNCA m = new MinistNCA(labels, images);

         boolean prevStateLoaded = false;

/*
        if (saveto.exists()){
            try {
                DataInput input = new DataInputStream(new BufferedInputStream(new FileInputStream(saveto)));
                m.rbm.load(input, m.layerFactory);
                prevStateLoaded = true;

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

*/

        if (!prevStateLoaded) {
            int numIterations = 50;
//          int numIterations = 1000; revisited

            m.rbm.setLayerFactory(m.layerFactory).
                    addLayer(dr.rows * dr.cols, false).
                    addLayer(500, false).
                    addLayer(500, false).
                    addLayer(2000, false).
                    addLayer(30, false).
                    build();

            System.err.println("Training level 1");
            m.learn(numIterations, false, 1);
            System.err.println("Training level 2");
            m.learn(numIterations, false, 2);
            System.err.println("Training level 3");
            m.learn(numIterations, false, 3);
            System.err.println("Training level 4");
            m.learn(numIterations, false, 4);

        }
    }

    public  void pretraining () {

        boolean prevStateLoaded = false;

/*
        if (saveto.exists()){
            try {
                DataInput input = new DataInputStream(new BufferedInputStream(new FileInputStream(saveto)));
                m.rbm.load(input, m.layerFactory);
                prevStateLoaded = true;

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

*/

        if (!prevStateLoaded) {
            int numIterations = 50;
//          int numIterations = 1000; revisited

            rbm.setLayerFactory(layerFactory).
                    addLayer(dr.rows * dr.cols, false).
                    addLayer(500, false).
                    addLayer(500, false).
                    addLayer(2000, false).
                    addLayer(30, false).
                    build();

            System.err.println("Training level 1");
            learn_epoch(numIterations, 100, false, 1);
            learn(numIterations, false, 1);
            System.err.println("Training level 2");
            learn_epoch(numIterations, 100, false, 2);
            System.err.println("Training level 3");
            learn_epoch(numIterations, 100, false, 3);
            System.err.println("Training level 4");
            learn_epoch(numIterations, 100, false, 4);


           /* System.err.println("Training level 1");
            learn(numIterations, false, 1);
            System.err.println("Training level 2");
            learn(numIterations, false, 2);
            System.err.println("Training level 3");
            learn(numIterations, false, 3);
            System.err.println("Training level 4");
            learn(numIterations, false, 4);

*/
        }
    }


    public static void test(MinistNCA m) {
        double numCorrect = 0;

        double numWrong = 0;
        double numAlmost = 0.0;

//        while (true) {
        for(int loop=0; loop<1000; loop++ ){
            MinstItem testCase = m.dr.getTestItem();

            Iterator<Tuple> it = m.evaluate(testCase);

            float[] labeld = new float[10];

            for (int i = 0; i < 2; i++) {
                Tuple t = it.next();

                for (int j = (t.visible.size() - 10), k = 0; j < t.visible.size() && k < 10; j++, k++) {
                    labeld[k] += t.visible.get(j);
                }
            }

            float max1 = 0.0f;
            float max2 = 0.0f;
            int p1 = -1;
            int p2 = -1;

            System.err.print("Label is: " + testCase.label);


            for (int i = 0; i < labeld.length; i++) {
                labeld[i] /= 2;
                if (labeld[i] > max1) {
                    max2 = max1;
                    max1 = labeld[i];

                    p2 = p1;
                    p1 = i;
                }
            }

            System.err.print(", Winner is " + p1 + "(" + max1 + ") second is " + p2 + "(" + max2 + ")");
            if (p1 == Integer.valueOf(testCase.label)) {
                System.err.println(" CORRECT!");
                numCorrect++;

            } else if (p2 == Integer.valueOf(testCase.label)) {
                System.err.println(" Almost!");
                numAlmost++;
            } else {
                System.err.println(" wrong :(");
                numWrong++;
            }

            System.err.println("Error Rate = " + ((numWrong / (numAlmost + numCorrect + numWrong)) * 100));

        }
    }

    public static void finetuning(File labels, File images) {
    }

    public void save_to_file(String s) {

        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(s)));
            oos.writeObject(this);
            oos.flush();;
            oos.close();
            System.out.println("output: " + s);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    public void error_test() {

        float[][] matrix = new float[5000][5000];
        float p_sum = 0;

        eval_configuration config = new eval_configuration();

        Index_for_mini_batch index_for_mini_batch = new Index_for_mini_batch().invoke(5000);
        ArrayList<Fine_tuning_tuple> minibatch_Set = index_for_mini_batch.getMinibatch_set();
        Map<String, List<Fine_tuning_tuple>> minibatch_Map = index_for_mini_batch.getMinibatch_map();
        p_sum = pre_compute(matrix, minibatch_Set);

        config.matrix=matrix;
        config.minibatch_Map=minibatch_Map;
        config.minibatch_Set = minibatch_Set;
        config.sum = p_sum;

        float O_NCA =  evaluate_O_nca(config);
        System.out.println("O_NCA: " + O_NCA);

//        standard_back_prop(rbm, minibatch_Set, config, 0.1f);

    }

    private void standard_back_prop(StackedRBM rbm, ArrayList<Fine_tuning_tuple> minibatch_set, eval_configuration config, float alpha) {

        for (Fine_tuning_tuple tuple : minibatch_set) {
            float partial_derivate = eval_partial_derivate(tuple, config);

        }




    }

    private float evaluate_O_nca(eval_configuration config) {
        float result = evaluate_O_nca(config.matrix, config.sum, config.minibatch_Set, config.minibatch_Map);
        config.O_NCA = result;
        config.precomputed=true;
        int size = config.minibatch_Set.size();
        float[][] matrix = config.matrix;
        float[][] matrix_2 = new float[size][size];

        float p_sum = config.sum;


        for (int i = 0; i < size; i++) {
            for (int j = i+1; j < size; j++) {
                matrix_2[i][j] = (float) Math.sqrt(matrix[i][j]);
                matrix_2[j][i] = (matrix[j][i])/p_sum-matrix[j][i];
            }
        }

        config.matrix = matrix_2;

        return result;
    }


    private float pre_compute(float[][] matrix, ArrayList<Fine_tuning_tuple> minibatch_Set) {

        float p_sum=0;
        int size = minibatch_Set.size();

        for (int i = 0; i < size; i++) {
            for (int j = i+1 ; j < size; j++) {
                Fine_tuning_tuple t_i = minibatch_Set.get(i);
                Fine_tuning_tuple t_j = minibatch_Set.get(j);

                matrix[i][j] = dist_square(t_i, t_j);
                matrix[j][i] = (float) Math.exp(-1 * matrix[i][j]);
                p_sum += matrix[j][i];
            }
        }
        return p_sum;
    }

    private float evaluate_O_nca(float[][] matrix, float p_sum, ArrayList<Fine_tuning_tuple> minibatch_Set, Map<String, List<Fine_tuning_tuple>> minibatch_Map) {

        float o_NCA =0;
        int size = minibatch_Set.size();
        for (int i = 0; i < size; i++) {

            Fine_tuning_tuple t_i = minibatch_Set.get(i);
            List<Fine_tuning_tuple> list = minibatch_Map.get(t_i.label);

            float p_t_i = 0;

            for (Fine_tuning_tuple tuple : list) {

                int j = minibatch_Set.indexOf(tuple);

                if(i==j)
                    continue;

                else if (i>j)
                    p_t_i += matrix[i][j];
                else
                    p_t_i += matrix[j][i];

/*
                float test = (float) Math.exp(-1 * dist_square(t_i, tuple));
                System.out.println(test);
*/
            }

            o_NCA += p_t_i;

        }

        o_NCA /= p_sum;

        return o_NCA;
    }

    private float dist_square(Fine_tuning_tuple t_i, Fine_tuning_tuple t_j) {

        float res = 0;
        float[] data_i = t_i.data;
        float[] data_j = t_j.data;

        for (int i = 0; i < 30; i++)
            res += (data_i[i] - data_j[i])*(data_i[i] - data_j[i]);

        return res;


    }

    private class Index_for_mini_batch {
        private Map<String, List<Fine_tuning_tuple>> minibatch_map;
        private ArrayList<Fine_tuning_tuple> minibatch_set;

        public Map<String, List<Fine_tuning_tuple>> getMinibatch_map() {
            return minibatch_map;
        }

        public ArrayList<Fine_tuning_tuple> getMinibatch_set() {
            return minibatch_set;
        }

        public Index_for_mini_batch invoke(int mini_batch_size) {
            minibatch_map = new HashMap<String, List<Fine_tuning_tuple>>();
            minibatch_set = new ArrayList<Fine_tuning_tuple>();

            Iterator<MinstItem> iter = dr.training_set.iterator(); //TODO
            for (int loop = 0; loop < mini_batch_size; loop++) {

                MinstItem trainingCase = iter.next();

                Layer output = feedfoward(trainingCase);

                Fine_tuning_tuple tuple = new Fine_tuning_tuple();
                {
                    tuple.label= trainingCase.label;
                    tuple.data = new float[30];
                }

                for (int i = 0; i < 30; i++)
                    tuple.data[i] = output.get(i);

                List<Fine_tuning_tuple> list =  minibatch_map.get(tuple.label);
                if(list == null)
                {
                    list =  new ArrayList<Fine_tuning_tuple>();
                    minibatch_map.put(tuple.label, list);
                }

                list.add(tuple);
                minibatch_set.add(tuple);
//                System.out.println(loop);

            }
            return this;
        }
    }


    private f_and_d_output f_and_d(Fine_tuning_tuple args, eval_configuration config) {

        f_and_d_output f_and_d_output_1 = new f_and_d_output();

        if(config.precomputed) evaluate_O_nca(config);

        float derivate = eval_partial_derivate(args, config);



        return null;
    }

    private float get_p (float[][] matrix, int i , int j) {
        return (i>j) ? matrix[i][j] : matrix[j][i];
    }

    private float get_d (float[][] matrix, int i , int j) {
        return (i>j) ? matrix[j][i] : matrix[i][j];
    }

    private float eval_partial_derivate(Fine_tuning_tuple tuple_a, eval_configuration config) {
        if(config.precomputed) evaluate_O_nca(config);

        float[][] matrix = config.matrix;
        float[][] matrix_2 = config.matrix_2;
        int size = config.minibatch_Set.size();
        int index_a = config.minibatch_Set.indexOf(tuple_a);
        float t2, t4;
        t2 = t4 = 0;
        List<Fine_tuning_tuple> list_same_label_a = config.minibatch_Map.get(tuple_a.label);

        //we don't need to compute t1, t3 since t1 = - 1 * t3 ; thus t1+t2+t3+t4 \ t1 + t4

/*
        //t1 eval
        for (int index_b = 0; index_b < list_same_label.size(); index_b++) {
            Fine_tuning_tuple tuple_b = list_same_label.get(index_b);
            if (tuple_a.equals(tuple_b)) continue;
            t1 += get_p(matrix, index_a, index_b) * get_p(matrix, index_a, index_b);
        }
        //t1 eval
*/


        //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval

        float t2_internal = 0;
        for (int z = 0; z < size; z++)
            if(index_a != z) t2_internal += get_p(matrix_2, index_a, z)*get_d(matrix_2, index_a, z);

        for (int index_b = 0; index_b < list_same_label_a.size(); index_b++) {
            if (tuple_a.equals(index_b)) continue;
            t2 += get_p(matrix_2, index_a, index_b) * t2_internal;
        }

        //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval //t2 eval

        //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval

        for (int index_l = 0; index_l < size; index_l++) {
            if(index_a==index_l) continue;
            float t4_internal = 0; //p_lq
            Fine_tuning_tuple tuple_l = config.minibatch_Set.get(index_l);
            List<Fine_tuning_tuple> list_same_label_l = config.minibatch_Map.get(tuple_l.label);

            for (int index_q = 0; index_q < list_same_label_l.size(); index_q++)
                t4_internal += get_p(matrix_2, index_l, index_q);

            t4 += t4_internal * get_p(matrix_2, index_l, index_a) * get_d(matrix_2, index_l, index_a);

        }

        //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval //t4 eval

    return Float.parseFloat(null);
    }


    ///////////////////////////

    private void scala_mul(SimpleRBM s, float v) {

        Layer bias = s.biasHidden;
        for (int i = 0; i < bias.size(); i++)
            bias.set(i, bias.get(i) * v);

        Layer[] weight = s.weights;

        for (Layer layer : weight) {
            for (int i = 0; i < layer.size(); i++) {
                layer.set(i, layer.get(i) * v );
            }
        }

    }

    private void scale_and_add(SimpleRBM x, float scala, SimpleRBM rhs) {

        Layer bias_x = x.biasHidden;
        Layer bias_r = rhs.biasHidden;
        for (int i = 0; i < bias_x.size(); i++)
            bias_x.set(i, bias_x.get(i) + scala * bias_r.get(i) );

        Layer[] weight_x = x.weights;
        Layer[] weight_r = rhs.weights;

        for (int i = 0; i < weight_x.length; i++) {
            Layer layer_x = weight_x[i];
            Layer layer_r = weight_r[i];
            for (int j = 0; j < layer_x.size(); j++) {
                layer_x.set(j, layer_x.get(j) + scala * layer_r.get(i));
            }
        }
    }
/*


    public void maximize (SimpleRBM X, Fine_tuning_tuple instance, eval_configuration config, int length){

        float _int = 0.1f;
        float ext = 3.0f;
        float _max = 20;
        float ratio = 10;
        float sig = 0.1f;
        float rho = sig/2;
        float red = 1.0f;

        int i=0;
        int is_failed = 0;

        f_and_d_output f0df0 = f_and_d(instance, config);
        float f0 = f0df0.f_out;
        SimpleRBM df0 = f0df0.derivative_out;

        float FX = f0; //71
        int fIter = 0; //72
        i = i + (length<0 ? 1: 0); //73
        s = -1 * df0; //74
        float d0 = - dot(s.T, s); //74

        float x3 = (float) (red / (1.0 - d0)); //75

        while (i < Math.abs(length)){ //77

            i = i + (length > 0 ? 1: 0); //78

            SimpleRBM X0 = X;
            float F0 = f0;
            SimpleRBM dFo = df0; //80

            int M;

            if (length>0)
                M = Integer.MAX_VALUE;
            else
                M = Math.min(Integer.MAX_VALUE, -1 * length -i);  //81

            float A, B, Z;
            float x1, x2, f1, f2, f3, d1, d2, d3;
            SimpleRBM df3;
                x2 = 0;
            while (true){ //83
                f2 = f0;
                d2 = d0;
                f3 = f0;
                df3 = df0;

                boolean success = false; //85
                while(!success && M > 0){
                    //TODO
                }


                if(f3 < F0) {
                    X0 = X + x3 * s;
                    F0 = f3;
                    df0 = df3; //96
                }

                d3 = df3.dot(s); //97

                if (d3 > sig * d0 || f3 > f0+x3*rho*d0 || M == 0) //98
                    break; //99

                x1 = x2; f1 = f2; d1 = d2;  //101 TODO
                x2 = x3; f2 = f3; d2 = d3 ; //102 TODO
                A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1); //103
                B = 3*(f2-f1)-(2*d1+d2)*(x2-x1); //104
                Z = (float) (B+Math.sqrt(B*B-A*d1*(x2-x1)));

                if(Z != 0.0f) {
                    x3 = ((x1 - (d1 * (x2 - x1) * (x2 - x1))) ) / Z; //105
                }
                else{
                    x3 = Int.in
                }





            }




        }



    }
*/
 /*   public void go (SimpleRBM X, Fine_tuning_tuple instance, eval_configuration config, int length){

//      int length=NSEARCHES;

        float _int = 0.1f;      // don't reevaluate within 0.1 of the limit of the current bracket
        float ext = 3.0f;       // extrapolate maximum 3 times the current bracket
        float _max = 20;        // max 20 function evaluations per line search
        float RATIO = 100;      // maximum allowed slope ratio

        float sig = 0.5f;       // RHO and SIG are the constants in the Wolfe-Powell conditions
        float rho = 0.01f;      // a bunch of constants for line searches


       //if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end 에 해당하는 것으로 보이는        int red=1;  //TODO:

        int red = 1;
        int i = 0;                          // zero the run length counter
        boolean ls_failed = false;        // no previous line search has failed
        boolean success;                   // TODO: ??? 없던 내용
        float f0,f1,f2,f3;                // TODO: ??? 없던 내용

        f_and_d_output f_and_d_output_1;

        SimpleRBM df0,df1,df2;                      // TODO: ??
        SimpleRBM tmp;                              // TODO: ??
        float d1,d2,d3;
        float z1,z2,z3;
        float A,B;
        float limit;
        SimpleRBM s;                                // TODO: ??
        SimpleRBM X0;                               // TODO: ??

        int M;
//        float realmin = std::numeric_limits<float>::min(); // TODO: ??
        //fX = []; list of function values
        // f1 = f(X,args);
        // df1 = d(X,args);

        f_and_d_output_1  = f_and_d(instance,config);
        df1 = f_and_d_output_1.derivative_out;
        f1 = f_and_d_output_1.f_out;
        // get function value and gradient
        // count epochs?!
        // printf("Changin s (51)");

        //    s = df1*(-1.0);                        // search direction is steepest

        s = df1;
        scala_mul(s, -1.0f);

        d1 = -s.dot(s);         // this is the slope //TODO
        z1 = red/(1-d1);   // initial step is red/(|s|+1)

        while (i < length){
            //printf("IN Linesearch %d ",i);
            // while not finished
            i = i + 1;          // count iterations?!

            X0 = X; f0 = f1; df0 = df1;  // make a copy of current values
            //      printf("Changing X (61). using rate %f\n",z1);

            //X = X + s*z1;
            scale_and_add(X,z1,s);

            // begin line search
            // f2 = f(X,args);
            // df2 =d(X,args);
            f_and_d(X,instance,df2,f2);

            d2 = df2.dot(s);
            f3 = f1; d3 = d1; z3 = -z1;
            // initialize point 3 equal to point 1
            if (length>0)
                M = _max;
            else
                M = fminf(_max, -length-i);

            success = false; limit = -1;                     // initialize quanteties

            ///new
            while (true){ //try the linesearch for upto 20 function iters.
                while( ((f2 > f1+z1*rho*d1) | (d2 > -sig*d1)) & (M > 0) ){
                    limit = z1;
                    // tighten the bracket
                    if (f2 > f1)
                        z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);        // quadratic fit
                    else{
                        A = 6*(f2-f3)/z3+3*(d2+d3);           // cubic fit
                        B = 3*(f3-f2)-z3*(d3+2*d2);
                        z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;  // numerical error possible - ok!
                    }
                    if (isnan(z2) | isinf(z2))
                        z2 = z3/2;   // if we had a numerical problem then bisect

                    z2 = fmaxf(fminf(z2, _int*z3),(1-_int)*z3);//don't accept close to limit
                    z1 = z1 + z2;        // update the step
                    //printf(" step %f ",z2);

                    //	  X = X + s*(z2);
                    X.scale_and_add(z2,s);

                    //  f2 = f(X,args);
                    //  df2 = d(X,args);
                    f_and_d(X,instance,df2,f2);
                    M = M - 1; // count epochs?!
                    d2 = df2.dot(s);
                    z3 = z3-z2;  // z3 is now relative to the location of z2
                }
*//*
		if(f2 < f1){
	  success = true;
	  break;
	}
	else *//*
                if ( (f2 > f1+z1*rho*d1) || (d2 > -sig*d1)){
                    //printf("Target: %f\n",f1+z1*rho*d1);
                    break;                         // this is a failure
                }
                else if (d2 > sig*d1){
                    success = true;
                    break;
                }                       // success
                else if( M == 0){
                    System.out.println("Leaving linesearch because too many func evals");
                    break;         // failure
                }
                //	printf("f2: %f target f: %f, d2: %f target d: %f d1: %f",f2,f1+z1*rho*d1,d2,sig*d1,d1);
                A = 6*(f2-f3)/z3+3*(d2+d3);    // make cubic extrapolation
                B = 3*(f3-f2)-z3*(d3+2*d2);
                z2 = (float) (-d2*z3*z3/(B+Math.sqrt(B*B-A*d2*z3*z3))); // num. error possible - ok! //TODO

                if ( isnan(z2) | isinf(z2) | z2 < 0){  // num prob or wrong sign?
                    if (limit < -0.5 )    // if we have no upper limit
                        z2 = z1 * (ext-1);  // the extrapolate the maximum amount
                    else
                        z2 = (limit-z1)/2;    // otherwise bisect
                }
                else if ((limit > -0.5) & (z2+z1 > limit))  // extraplation beyond max?
                    z2 = (limit-z1)/2;                    // bisect
                else if ((limit < -0.5) & (z2+z1 > z1*ext)) // extrapolation beyond limit
                    z2 = z1*(ext-1.0);     // set to extrapolation limit
                else if (z2 < -z3*_int)
                    z2 = -z3*_int;
                else if ((limit > -0.5) & (z2 < (limit-z1)*(1.0-_int)))//close to limit?
                    z2 = (limit-z1)*(1.0-_int);

                f3 = f2; d3 = d2; z3 = -z2;       // set point 3 equal to point 2
                //	printf("Changin X. (133) using step %f\n",z2);
                z1 = z1 + z2;

                //X = X + s*(z2);
                X.scale_and_add(z2,s);

                //	f2 = f(X,args);// update current estimates
                //	df2 =d(X,args);
                //	printf("(142) Step: %f\n",z2);
                f_and_d(X,instance,df2,f2);
                M = M - 1;  // count epochs?!
                d2 = df2.dot(s);
            }                 // end of line search

            if (success){         // if line search succeeded

                f1 = f2; //fX = [fX' f1]';
                //	printf("SUCCEDDED F val: %f\n",f1);
                //printf(" SUCCESS ON LINESEARCH %d;  Value %f\n", i, f1);
                //printf("Changing s (147)");

                //s = s*(  ( df2.dot(df2)- df1.dot(df2))/df1.dot(df1) ) + df2*(-1); //PR

                s *=  ( df2.dot(df2)- df1.dot(df2))/df1.dot(df1) ;
                s.scale_and_add( -1,df2);

                //	printf("Werd to big berd\n");
                tmp = df1; df1 = df2; df2 = tmp; //swap derivatives
                d2 = df1.dot(s);
                if (d2 > 0){
                    // printf("changing s (152)");// new slope must be negative

                    //s = df1*(-1);  // otherwise use steepest direction

                    s = df1;
                    s *= -1;

                    d2 = -s.dot(s);
                }

                z1 = z1 * fminf(RATIO, d1/(d2-realmin));  // slope ratio but max RATIO
                d1 = d2;
                ls_failed = false; // this line search did not fail
            }
            else{
                //printf(" FAIL ON SEARCH %d\n",i);
                X = X0; f1 = f0; df1 = df0;  // restore point from before failed line search
                if (ls_failed | i > length){ // line search failed twice in a row
                    break;             // or we ran out of time, so we give up
                }
                tmp = df1; df1 = df2; df2 = tmp;      // swap derivatives

                //	s = df1*(-1.0);          // try steepest
                s = df1;
                s *= -1.0;

                d1 = -s.dot(s);
                z1 = 1/(1-d1);
                ls_failed = true; // this line search failed
            }

        }
        _fail = ls_failed;
        printf("LS done\n");
        return X;


    }
*/
/*
    float poly_min(float c, float b, float r, float fr, float fail){
    *//*given constraints f(0) = c, f(r) = fr, f'(0) = b,
      return minimum of polynomial, or -0.1 if unbounded from below *//*
        float a = (fr - b*r - c)/(r*r);
        if (a <= 0){
            printf("Fail. falling back on %f\n",fail);
            return fail;
        }
        else{
            printf("Cool. using m=%f\n",-b/(2.0 * a));
            return -b/(2.0 * a);
        }
    }

    T search(T &p, U & args){
        bool success=false;
        int tries=0;

        float c = f(p,args);
        printf("Error unnorm: %f ",c);
        T dp = d(p,args);
        float b *//*= dp.squaredNorm()*//*;
        T ret;
        while (!success and tries<10){

            float r = pow(10,-tries)*(drand48()-1.0); //gaussian random
            T x1 = p + (dp*r);

            float fr = f(x1,args);

            float m = poly_min(c,b,r,fr, -0.0001);
            ret = p + (dp*m);
            float val = f(ret,args);
            printf("Try %d val %f\n",tries,val);
            if( val < c)
                success=true;
            tries++;
        }
        return ret;
    }

    T search(T &p, const T & dir, U & args){
        float c = f(p,args);
        printf("Error: unnorm: %f ",c);
        T dp = d(p,args);
        //  printf("Done with derivative\n");
        float b = dir.dot(dp);

        float r = (0.01)*(drand48()-1.0); //gaussian random
        T x1 = p + (dir*r);
        //   printf("About to X1. x1 weight size: %d bias sz %d\n",
        //	   x1.weights.size(),x1.biases.size());
        float fr = f(x1,args);

        float m = poly_min(c,b,r,fr, -0.0001);
        T ret = p + (dp*m);

        return ret;
    }*/

}


class f_and_d_output {

    public SimpleRBM derivative_out;
    public float f_out;
}


class Fine_tuning_tuple{
    public String label;
    public float[] data;
}

class pre_computation_output{
    public float[][] matrix;
    public float sum;
}

class eval_configuration{

    public eval_configuration() {
        this.precomputed = false;
    }

    public boolean  precomputed;
    public float[][] matrix;
    public float[][] matrix_2;
    public ArrayList<Fine_tuning_tuple> minibatch_Set;
    public Map<String, List<Fine_tuning_tuple>> minibatch_Map;
    public float sum;
    public float O_NCA;

}