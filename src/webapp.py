"""
This file contains all functions related to the webserver, i.e. the code to run the server, display the html sites and all callback.
It holds the global DATA which contains metadata as well as the actual data numpy matrix.
It uses the Dash-based html code from views/ui, and the controllers implemented in controllers/.
"""
import os
from multiprocessing import Process, Queue
import multiprocessing as mp

# Dash server, html and core components as well as bootstrap components and
# callback parameters
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from plotly import graph_objects as go

# controllers to select, preprocess and analyze data.
from controllers.select import (update_electrode_selection, convert_to_jpeg,
                                max_duration, str_to_mus, update_time_window)
from controllers.preproc import frequency_filter, downsample, filter_el_humming
from controllers.analysis.analyze import (animate_amplitude_grid, show_psds,
                                 show_moving_averages, detect_peaks_amplitude,
                                 show_spectrograms,
                                 show_periodic_aperiodic_decomp,
                                 detect_events_moving_dev)

## Code used to import data into a Data object, see model/Data.py
from controllers.io.import_mcs_256 import mcs_256_import #extract_baseline_std
from controllers.io.import_mcs_cmos import mcs_cmos_import

# Dash-wrapped html code for the UI
from ui.nav import navbar, nav_items
from ui.importer import importer, build_import_infos
from ui.select import select, no_data
from ui.preproc import preproc
from ui.analyze import analyze

# Plots using Plotly for selection and matplotlib for everything else
from views.electrode_grid import draw_electrode_grid

# setup for the server and initialization of the data global
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
content = html.Div(id="page-content")
app.layout = html.Div([dcc.Location(id="url"), navbar, content])
DATA_PATH = None
DATA = None
RESULT = None
BL_PROC = None
BL_QUE = None


@app.callback(
        Output("page-content", "children"), Output("nav", "children"),
        [Input("url", "pathname")]
        )
def render_page_content(pathname: str) -> html.Div:
    """
    Function that changes the html contents of the webpage depending on the
            specified URL

        @param pathname: string provided by browsers URL bar on user input;
                Contains what comes after localhost:8080

        @return the page contents as implemented in views/ui/. The navbar
                is only displayed when not on the home/import screen.
    """
    if pathname == "/" or DATA is None:
        return importer, None

    if pathname == "/select" or RESULT is None:
        grid = draw_electrode_grid(DATA)
        return select(grid), nav_items

    if pathname == "/preproc":
        return preproc, nav_items

    if pathname == "/analyze":
        return analyze, nav_items

    # If the user tries to reach a different page, return a 404 message
    return html.Div(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
                ],
            className="p-3 bg-dark rounded-3",
            ), nav_items


@app.callback(Output("import-feedback", "children"),
              Input("import-submit-input-file-path", "n_clicks"),
              State("import-input-file-path", "value"),
              State("import-radios", "value"),
              prevent_initial_call=True)
def import_file(_: int, input_file_path: str, file_type: int) -> list[html.Div]:
    """
    Used on home/import screen.

    Loads data from file into a Data object, stored in the DATA global
            variable, see model/Data.py using the importers provided in
            model/io.
    So far only the 252 channel MEA by MultiChannel Systems is supported.

        @param input_file_path: path to the file containing data
        @param file_type: the type of the input file, used to choose the
                apropriate importer e.g. MCS 256.

        @return feedback if the import was successful. If so metadata shall be
                displayed, if not an error message is shown.
    """
    if cond_input_file_path is None or file_type is None:
        return build_import_infos("Please enter a file path!", success=False)

    global DATA, DATA_PATH
    import_que = Queue()

    if file_type == 0:
        proc_import = Process(target=mcs_256_import, args=(cond_input_file_path, import_que))
    elif file_type == 1:
        proc_import = Process(target=mcs_cmos_import, args=(cond_input_file_path, import_que))
    else:
        raise IOError("Only Multi Channel System H5 file format is supported"
                      "so far.")

    proc_import.start()

    DATA, info = import_que.get()

    proc_import.join()

    success = True if DATA is not None else False
    DATA_PATH = input_file_path
    feedback = build_import_infos(info, success=success)

    return feedback


@app.callback(Output("select-mea-setup-img", "src"),
              Input("select-image-file-path", "value"),
              prevent_initial_call=True)
def select_set_recording_image(image: str) -> str:
    """
    Used on select screen.

    Takes a file path and shows that image next to the electrode grid, s.t.
            the user can see where the probe is located on the MEA and what
            electrode is where in the probe.

        @param image, the file path to the image to be displayed in the
                select screen
        @param previous image. used as the callback is also executed on
                loading the site and then it should just display the default
                image.

        @return the image loaded from the file path specified, converted to
                jpeg to assure compatibility across browsers (e.g. if the
                input is in tiff format.
    """

    img = convert_to_jpeg(image)

    return img


@app.callback(Output("select-start", "placeholder"),
              Output("select-stop", "placeholder"),
              Input("select-time", "children"))
def set_time_span(_) -> str:
    """
    Used on select screen.

    sets the placeholder for the select time window input fields, such that
            the user can see the min value (always 0) and the max value for
            the time in s:ms:mus (instead of just microseconds)

        @param _: used as dash callbacks have to have an input and output.

        @return the minimum and maximum possible time in s:ms:mus
    """
    s_start, ms_start, mus_start = 0, 0, 0
    s_end, ms_end, mus_end = max_duration(DATA)

    return f"{s_start}:{ms_start:03}:{mus_start:03}", (f"{s_end}:{ms_end:03}"
            ":{mus_end:03}")


@app.callback(Output("select-electrode-grid", "figure"),
              Output("select-electrode-grid", "selectedData"),
              Output("select-electrode-grid", "clickData"),
              Input("select-electrode-grid", "selectedData"),
              Input("select-electrode-grid", "clickData"),
              prevent_initial_call=True)
def select_update_selection_and_grid(selected_electrodes: str,
                                     clicked_electrode: str
                                     ) -> tuple[go.Figure, None, None]:
    """
    Used on select screen.

    If an electrode is clicked or selected by box or lasso on the electrode
            grid, the corresponding row in the data matrix is added to the
            list of selected electrodes and the electrode is toggled from
            unselected/red to selected/green or vice versa on the grid.

        @param selected_electrodes: The electrodes that are selected via box
                or lasso select.
        @param clicked_electrode: The electrode that was selected by a click.

        @return the updated electrode grid to be shown
    """
    update_electrode_selection(DATA, selected_electrodes,
                               clicked_electrode)
    grid = draw_electrode_grid(DATA)


    return grid, None, None


@app.callback(Output("select-output-dummy", "children", allow_duplicate=True),
              Input("select-show-signals", "n_clicks"),
              State("select-start", "value"),
              State("select-stop", "value"),
              prevent_initial_call=True)
def select_plot_raw(_: int, t_start: str, t_end: str) -> None:
    """
    Used on select screen.

    Plots the raw signals of the selected electrodes in the given time window.

        @param t_start: the start of the selected time windw in s:ms:mus
        @param t_end: the end of the selected time window in s:ms:mus

        @return A dummy as dash callbacks require an output. The plotting is
                done in a separate process by matplotlib (as Dash plots are
                only suitable for small amounts of data)
    """
    # FIXME come back here after viz is fixed
    # converts the start and end time from s:ms:mus to mus
    update_time_window(DATA, t_start, t_end)

    #proc = Process(target=plot_in_grid, args=('time_series', signals, data.selected_rows, \
            #        names_selected_sorted, data.sampling_rate, start, end))
    #proc.start()
    #proc.join()

    return None


@app.callback(Output("select-output-dummy", "children", allow_duplicate=True),
              Input("select-apply", "n_clicks"),
              State("select-start", "value"),
              State("select-stop", "value"),
              prevent_initial_call=True)
def select_apply(_: int, t_start: str, t_stop: str) -> None:
    """
    Used by select screen.

    Discards all but the selected rows, and all columns that are outside of
            the selected time window.

        @param t_start: start of the time window in s:ms:mus
        @param t_stop: end of the time window in s:ms:mus
        @param prev_button: used to show the default button when the callback
                is executed on load

        @retrun a next button to get to the preprocessing page.
    """
    global RESULT
    update_time_window(DATA, t_start, t_stop)
    RESULT = create_result(DATA)

    return None


@app.callback(Output("preproc-fltr-result", "children"),
              Input("preproc-fltr-apply", "n_clicks"),
              State("preproc-fltr-lower", "value"),
              State("preproc-fltr-upper", "value"),
              State("preproc-fltr-type", "value"),
              prevent_initial_call=True)
def preproc_filter(_: int, lower: str, upper: str, ftype: str) -> html.Div:
    """
    Used by the preprocessing screen.

    Filters all data rows with a butterworth band pass or top filter.
    For a bandpass filter the frequencies below lower and above upper are discarded.
    For a bandstop filter the frequencies between lower and upper are discarded.

    @param lower: lower pass or stop frequency limit
    @param upper: higher pass or stop frequency limit.
    @param ftype:  wether to use a bandpass or a band stop filter.

    @return A banner indicating that the filter was applied.
    """
    frequency_filter(RESULT, ftype, float(lower), float(upper), bool(ftype))

    return dbc.Alert("Successfully applied bandstop filter", color="success")


@app.callback(Output("preproc-dwnsmpl-result", "children"),
              Input("preproc-dwnsmpl-apply", "n_clicks"),
              State("preproc-dwnsmpl-rate", "value"),
              prevent_initial_call=True)
def preproc_downsample(_, sampling_rate: str) -> html.Div:
    """
    Used by the preprocessing screen.

    Decimates the signal to contain as many data points as the signal would
            have if it was sampled at rate fs.
    Uses scipy.signal.decimate, which avoids aliasing.

        @param clicked: button to cause the application of the downsampling.
        @param fs: new sampling rate.

        @return a banner indicating if the downsampling was applied
    """
    downsample(RESULT, int(sampling_rate))

    return dbc.Alert("Successfully downsampled", color="success")


@app.callback(Output("preproc-humming-result", "children"),
              Input("preproc-humming-apply", "n_clicks"),
              prevent_initial_call=True)
def preproc_humming(_) -> html.Div:
    """
    Used by preprocessing screen.

    Removes noise caused by the electrical system's frequency which is 50 Hz
            in Europe. I.e. removes the 50 Hz component from the signal

        @param clicked: button that causes the fitering to be applied.

        @return a banner indicating if the filter was applied
    """
    filter_el_humming(RESULT)

    return dbc.Alert("Successfully removed electrical humming",
            color="success")


@app.callback(Output("analyze-animate-play", "n_clicks"),
              Input("analyze-animate-play", "n_clicks"),
              State("analyze-animate-fps", "value"),
              State("analyze-animate-slow-down", "value"))
def analyze_amplitude_animation(clicked: int, fps: str, slow_down, t_start, t_stop):
    """
    Used by analyze screen.

    Draws the electrode grid as on the MEA and color codes the current
    absolute amplitude binned to the frame per second rate specified
    in fps and creates a video (mp4/h264) from it of the specified
    time window.

        @param clicked: Button causing the generation of the video
        @param fps: How many frames shall be generated per second. the lower,
                    the wider the bins, i.e. the worse the temporal
                    resolution of the video but the smaller the video size.
        @param t_start: Where the video shall begin. if not specified
                        (i.e. None), 0 is choosen.
        @param t_stop: Where the video shall end. If not specified, the
                       duration of the signal is chosen.

        @return sets the button clicks back to 0, done because dash callbacks
                       have to have an output
    """
    if clicked is not None and clicked > 0:
        new_sr = fps / slow_down
        bins = bin_amplitude(RESULT, fps, slow_down)

    return 0


@app.callback(Output("analyze-psd", "n_clicks"),
              Input("analyze-psd", "n_clicks"))
def analyze_psds(clicked):
    """
    used by analyze screen.

    Computes the power spectral densities for all selected rows and plots the \
       #            results.
    """
    if clicked is not None and clicked > 0:
        show_psds(DATA)

    return 0


@app.callback(Output("analyze-aperiodic-periodic", "n_clicks"),
              Input("analyze-aperiodic-periodic", "n_clicks"))
def analyze_periodic_aperiodic(clicked):
    """
    Used by analyze screen.

    Computes the PSDs for selected electrodes and then separates the periodic
                   from the aperiodic part and shows the results.
    """
    if clicked is not None and clicked > 0:
        show_periodic_aperiodic_decomp(DATA)

    return 0


@app.callback(Output("analyze-spec", "n_clicks"),
              Input("analyze-spec", "n_clicks"))
def analyze_spectrograms(clicked):
    """
    used by analyze screen.

    Computes the spectrogram for all selected rows and plots the results.
    """
    if clicked is not None and clicked > 0:
        show_spectrograms(DATA)

    return 0


@app.callback(Output("analyze-peaks-ampl", "n_clicks"),
              Input("analyze-peaks-ampl", "n_clicks"),
              State("analyze-peaks-ampl-loc-thresh", "value"),
              State("analyze-peaks-ampl-glob-thresh", "value"))
def analyze_peaks_ampl(clicked, loc_thresh_factor, glob_thresh_factor):
    """
    used by analyze screen.

    Detects peaks in the signal by the absolute amplitude with a threshold \
       #            depending on the standard deviation of a baseline signal or half \
       #            the signal std.
    """
    if clicked is not None and clicked > 0:
        if loc_thresh_factor is not None and glob_thresh_factor is not None:
            detect_peaks_amplitude(DATA, True, STD_BASE, float(loc_thresh_factor), float(glob_thresh_factor))
        elif loc_thresh_factor is not None:
            detect_peaks_amplitude(DATA, True, STD_BASE, float(loc_thresh_factor))
        elif glob_thresh_factor is not None:
            detect_peaks_amplitude(DATA, True, STD_BASE, global_std_factor=float(glob_thresh_factor))
        else:
            detect_peaks_amplitude(DATA, True, STD_BASE)

    return 0


@app.callback(Output("analyze-events-stats", "children"),
              Output("analyze-events", "n_clicks"),
              Input("analyze-events", "n_clicks"),
              State("analyze-events-method", "value"),
              State("analyze-events-thresh", "value"),
              State("analyze-events-export", "value"),
              State("analyze-events-fname", "value"),
              )
def analyze_events_moving_dev(clicked, method, thresh_factor, export, fname):
    """
    Used by analyze screen.

    Detects events/bursts by computing the moving average with a large window \
       #            and a threshold.
    """
    export = len(export) > 0
    res = None
    if clicked is not None and clicked > 0:
        std = STD_BASE_MV_STD if method == 1 else STD_BASE_MV_MAD

        if thresh_factor is None:
            res = detect_events_moving_dev(DATA, method, STD_BASE_MV_STD, export=export, fname=fname)
        else:
            res = detect_events_moving_dev(DATA, method, STD_BASE_MV_STD, float(thresh_factor), export=export, fname=fname)

    return res, 0


if __name__ == "__main__":
    print("LFP Toolbox")
    mp.set_start_method('spawn')

    if not os.path.exists(os.path.join(os.getcwd(), "plots")):
        os.mkdir(os.path.join(os.getcwd(), "plots"))

    HOST = "localhost"
    PORT = 8080

    app.run_server(
            host=HOST,
            port=PORT,
            debug=True
            )
