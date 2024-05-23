import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.Styler;

import java.util.ArrayList;
import java.util.List;

/**
 * Gaussian Blob
 *
 * Demonstrates the following:
 * <ul>
 * <li>ChartType.Scatter
 * <li>Series data as a Set
 * <li>Setting marker size
 * <li>Formatting of negative numbers with large magnitude but small differences
 */
public class ScatterPlot { // implements ExampleChart<XYChart>

    XYChart chart;
    List<Double> Xdata;
    List<Double> Ydata;
    String startingSeriesName;

    // TODO: Add functionality to plot multiple series and be able to see it

    public ScatterPlot(String title, String staringSeriesName) {

        Xdata = new ArrayList<>();
        Ydata = new ArrayList<>();
        this.startingSeriesName = staringSeriesName;

        chart = new XYChartBuilder().width(800).height(600).build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
        chart.getStyler().setChartTitleVisible(false);
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideSW);
        chart.getStyler().setMarkerSize(16);
    }


    public void addData(double x, double y) {
        Xdata.add(x);
        Ydata.add(y);
    }

    public void graph() {
        chart.addSeries(startingSeriesName, Xdata, Ydata);
       new SwingWrapper<XYChart>(chart).displayChart();
    }


//    public static void main(String[] args) {
//
//        ExampleChart<XYChart> exampleChart = new ScatterChart01();
//        XYChart chart = exampleChart.getChart();
//        new SwingWrapper<XYChart>(chart).displayChart();
//    }

//    @Override
//    public XYChart getChart() {
//
//        // Create Chart
//        XYChart chart = new XYChartBuilder().width(800).height(600).build();
//
//        // Customize Chart
//        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
//        chart.getStyler().setChartTitleVisible(false);
//        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideSW);
//        chart.getStyler().setMarkerSize(16);
//
//        // Series
//        List<Double> xData = new LinkedList<Double>();
//        List<Double> yData = new LinkedList<Double>();
//        Random random = new Random();
//        int size = 1000;
//        for (int i = 0; i < size; i++) {
//            xData.add(random.nextGaussian() / 1000);
//            yData.add(-1000000 + random.nextGaussian());
//        }
//        chart.addSeries("Gaussian Blob", xData, yData);
//
//        return chart;
//    }




}