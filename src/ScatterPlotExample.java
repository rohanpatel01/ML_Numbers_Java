import java.awt.Color;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

/*
    TODO: add functionality to plot multiple series and add them dynamically
 */


public class ScatterPlotExample extends JFrame {
    private static final long serialVersionUID = 6294689542092367723L;
    public XYSeriesCollection dataset;
    public XYSeries series1;

    public ScatterPlotExample(String title, String startingSeriesName) {
        super(title);

        dataset = new XYSeriesCollection();
        series1 = new XYSeries(startingSeriesName);

    }

    public void graph() {
        SwingUtilities.invokeLater(() -> {
            this.setSize(800, 400);
            this.setLocationRelativeTo(null);
            this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            this.setVisible(true);
        });

    }

//    public void addSeries() {
//
//    }

    // TODO: add support for which series we want to add the data to
    public void addXYData(double x, double y) {
        series1.add(x, y);
    }

    public void createChart() {
       // Create chart
        JFreeChart chart = ChartFactory.createScatterPlot(
                "Boys VS Girls weight comparison chart",
                "X-Axis", "Y-Axis", dataset);


        //Changes background color
        XYPlot plot = (XYPlot)chart.getPlot();
        plot.setBackgroundPaint(new Color(255,228,196));


        // Create Panel
        ChartPanel panel = new ChartPanel(chart);
        setContentPane(panel);

        dataset.addSeries(series1);
//        return dataset;
    }

}