package com.github.tjake.rbm.minst;

import com.github.tjake.rbm.BinaryLayer;
import com.github.tjake.rbm.Layer;
import com.github.tjake.rbm.SimpleRBM;
import com.github.tjake.rbm.Tuple;
import com.github.tjake.util.Utilities;

import java.awt.*;
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
        for (int i = 0; i < 28*28; i++) {

            item.data[i] = (Math.random() < v.get(i) ) ? 40: 0;
        }

        System.out.println();


/*
        Layer input = nca.layerFactory.create(item.data.length);

        for (int i = 0; i < item.data.length; i++)
            input.set(i, item.data[i]);

        input = new BinaryLayer(input);

        SimpleRBM s_rbm = nca.rbm.getInnerRBMs().get(0);

        Layer hidden_1  = s_rbm.activateHidden(input, s_rbm.biasHidden);
        Layer visible_2 = s_rbm.activateVisible(hidden_1, s_rbm.biasVisible);

        int size = visible_2.size();

        for (int i = 0; i < size; i++) {

            item.data[i] = (Math.random() > visible_2.get(i) ) ? 1: 0;
        }
*/

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
}
