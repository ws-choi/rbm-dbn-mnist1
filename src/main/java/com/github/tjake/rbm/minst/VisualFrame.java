package com.github.tjake.rbm.minst;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

/**
 * Created by wschoi on 2015-12-06.
 */
public class VisualFrame extends JFrame {

    VisualTest vt;
    JButton pretraining, load_data,reconstruct, test, save_to_file, load_from_file, error_test;
    File labels, images;

    public VisualFrame(MinistNCA nca, final File labels, final File images) {

        super("MINST Draw");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        vt = new VisualTest(nca);
        this.labels=labels;
        this.images=images;

        add(vt);

        pretraining = new JButton("pre_training");
        pretraining.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

//                vt.nca.pretraining();
            }
        });

        load_data = new JButton("load data");
        load_data.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {

                vt.loaddata();
            }
        });

        reconstruct = new JButton("recontruct");
        reconstruct.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                vt.reconstruct();
            }
        });



        test = new JButton("test");
        test.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                vt.test();
            }
        });

        save_to_file = new JButton("save to file");
        save_to_file.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                vt.save_to_file();
            }
        });

        load_from_file = new JButton("load from file");
        load_from_file.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                vt.load_from_file(labels, images);
            }
        });

        error_test = new JButton("Error Test");
        error_test.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                vt.error_test();
            }
        });

        JPanel bottom = new JPanel();
        bottom.add(pretraining);
        bottom.add(load_data);
        bottom.add(reconstruct);
        bottom.add(test);
        bottom.add(save_to_file);
        bottom.add(load_from_file);
        bottom.add(error_test);

        add(bottom, BorderLayout.SOUTH);

        pack();

        setLocationRelativeTo(null);


        int total = 0;
        for (int i = 0; i < 10; i++) {
            total += nca.dr.trainingSet.get(String.valueOf(i)).size();
        }
        System.out.println(total);
    }
}
