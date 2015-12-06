package com.github.tjake.rbm.minst;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Created by wschoi on 2015-12-06.
 */
public class VisualFrame extends JFrame {

    VisualTest vt;
    JButton load_data,reconstruct;

    public VisualFrame(MinistNCA nca) {


        super("MINST Draw");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        vt = new VisualTest(nca);
        add(vt);
        pack();

        setLocationRelativeTo(null);


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


        JPanel bottom = new JPanel();
        bottom.add(load_data, BorderLayout.WEST);
        bottom.add(reconstruct, BorderLayout.EAST);
        add(bottom, BorderLayout.SOUTH);
    }
}
