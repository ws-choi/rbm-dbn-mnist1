package com.github.tjake.rbm.minst;

import com.github.tjake.rbm.BinaryLayer;
import com.github.tjake.rbm.Layer;
import com.github.tjake.rbm.SimpleRBM;
import com.github.tjake.rbm.Tuple;
import com.github.tjake.util.Utilities;

import java.awt.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Iterator;

public class VisualTest extends Canvas{

    public static MinistNCA nca;
    private MinstItem item;
    private Iterator<Tuple> iter = null;
    public VisualTest(MinistNCA nca) {

        this.nca =  nca ;
        setSize(560, 560);



    }

    public void loaddata() {

        item = nca.dr.getTestItem();
        iter = null;
        repaint();;
    }

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        paintcomponent(g);
    }

    private void paintcomponent(Graphics g) {

        if(item != null){

            int[] data = item.data;

            for (int i1 = 0; i1 < data.length; i1++) {
                int i = data[i1];
                if(i>0)
                {
                    g.setColor(Color.black);
                    g.fillRect(20*(i1%28), 20*(i1/28), 20, 20);
                }
                else
                {
                    g.setColor(Color.WHITE);
                    g.fillRect(20*(i1%28), 20*(i1/28), 20, 20);
                }

            }


        }

    }

    Iterator<Tuple> evaluate() {


        Layer input = nca.layerFactory.create(item.data.length);

        for (int i = 0; i < item.data.length; i++)
            input.set(i, item.data[i]);

        return nca.rbm.getInnerRBMs().get(0).iterator(new BinaryLayer(input));
    }

    public void reconstruct() {

        if(iter ==null) {
            SimpleRBM s_rbm = nca.rbm.getInnerRBMs().get(0);
            Layer input = get_InputLayer();
            iter = s_rbm.iterator(input);
        }

        Tuple t = iter.next();

        Layer v = t.visible;

        float[] visible = BinaryLayer.fromBinary(t.visible);

        for (int i = 0; i < 28*28; i++) {

            item.data[i] = ( visible[i] > 40 ) ? 40: 0;
        }

        repaint();;


    }

    private Layer get_InputLayer() {
        Layer input = new Layer(item.data.length);
        for (int i = 0; i < item.data.length; i++) {
            input.add(i, item.data[i]);
        }
        input = new BinaryLayer(input);
        return input;
    }

    public void test() {
        nca.test(nca);
    }

    public void save_to_file() {
        nca.save_to_file("nca.obj");
    }

    public void load_from_file() {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File("nca.obj")));
            try {
                nca = (MinistNCA) ois.readObject();
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }

            ois.close();
            System.out.println("success");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
