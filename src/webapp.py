"""
This file contains all functions related to the webserver, i.e. the code to
run the server, display the html sites and all callback. It holds the global
DATA which contains metadata as well as the actual data numpy matrix. It uses
the Dash-based html code from views/ui, and the controllers implemented in
controllers/.
"""
import os
from multiprocessing import Process, Queue
import multiprocessing as mp

# Dash server, html and core components as well as bootstrap components and
# callback parameters
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
from plotly import graph_objects as go

# Code used to import data into a Data object, see model/Data.py
from controllers.io.import_mcs_256 import mcs_256_import
# controllers to select, preprocess and analyze data.
from controllers.select import (apply_selection, convert_to_jpeg,
                                update_electrode_selection, max_duration,
                                update_time_window)
from controllers.analysis.filter import (frequency_filter, downsample,
                                         filter_el_humming)
from controllers.analysis.analyze import (compute_snrs,
                                          compute_rms,
                                          compute_derivatives,
                                          compute_mv_avgs,
                                          compute_mv_mads,
                                          compute_envelopes,
                                          compute_entropies)
from controllers.analysis.activity import detect_peaks
from controllers.analysis.spectral import (compute_psds,
                                           compute_periodic_aperiodic_decomp,
                                           detrend_fooof,
                                           compute_spectrograms)
from controllers.analysis.network import (compute_xcorrs, compute_mutual_info,
                                          compute_transfer_entropy,
                                          compute_coherence,
                                          compute_granger_causality,
                                          compute_spectral_granger,
                                          compute_current_source_density)


# Dash-wrapped html code for the UI
from ui.nav import navbar, nav_items
from ui.importer import importer, build_import_infos
from ui.select import select, no_data, next_button
from ui.analyze import analyze, generate_table, TimeSeriesPlottable

# Plots using Plotly for selection and pyqtgraph everything else
from views.electrode_grid import draw_electrode_grid
from views.time_series import plot_time_series_grid

# setup for the server and initialization of the data global
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True,
           prevent_initial_callbacks="initial_duplicate")

content = html.Div(id="page-content")
app.layout = html.Div([dcc.Location(id="url"), navbar, content])
# DATA shared memory
DATA = None


# ================= Routing
@app.callback(Output("page-content", "children"), Output("nav", "children"),
              [Input("url", "pathname")])
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

    if pathname == "/select":
        grid = draw_electrode_grid(DATA)
        return select(grid), nav_items

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


# ============= Import
@app.callback(Output("import-feedback", "children"),
              Input("import-submit-input-file-path", "n_clicks"),
              State("import-input-file-path", "value"),
              State("import-radios", "value"),
              prevent_initial_call=True)
def import_file(_: int,
                input_file_path: str,
                file_type: int
                ) -> list[html.Div]:
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
    if input_file_path is None or file_type is None:
        return build_import_infos("Please enter a file path!", success=False)

    global DATA
    import_que = Queue()

    if file_type == 0:
        proc_import = Process(target=mcs_256_import,
                              args=(input_file_path, import_que))
#    elif file_type == 1:
#        proc_import = Process(target=mcs_cmos_import,
#                              args=(input_file_path, import_que))
    else:
        raise IOError("Only Multi Channel System H5 file format is supported"
                      "so far.")

    proc_import.start()

    DATA, info = import_que.get()

    proc_import.join()
    success = DATA is not None
    feedback = build_import_infos(info, success=success)

    return feedback


# ================== Select
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

    return (f"{s_start}:{ms_start:03}:{mus_start:03}",
            f"{s_end}:{ms_end:03}:{mus_end:03}")


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
def select_plot_raw(_: int, t_start: str, t_end: str) -> list:
    """
    Used on select screen.

    Plots the raw signals of the selected electrodes in the given time window.

        @param t_start: the start of the selected time windw in s:ms:mus
        @param t_end: the end of the selected time window in s:ms:mus

        @return A dummy as dash callbacks require an output. The plotting is
                done in a separate process by matplotlib (as Dash plots are
                only suitable for small amounts of data)
    """
    # converts the start and end time from s:ms:mus to mus
    update_time_window(DATA, t_start, t_end)
    plot_time_series_grid(DATA, selected=False)

    return []


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
    if len(DATA.selected_electrodes) == 0:
        return no_data
    update_time_window(DATA, t_start, t_stop)
    apply_selection(DATA)

    return next_button


# ====================== ANALYZE
# ======== Filter
@app.callback(Output("analyze-fltr-result", "children"),
              Input("analyze-fltr-apply", "n_clicks"),
              State("analyze-fltr-lower", "value"),
              State("analyze-fltr-upper", "value"),
              State("analyze-fltr-type", "value"),
              prevent_initial_call=True)
def analyze_filter(_: int, lower: str, upper: str, ftype: str) -> html.Div:
    """
    Used by the preprocessing screen.

    Filters all data rows with a butterworth band pass or top filter.
    For a bandpass filter the frequencies below lower and above upper are
        discarded.
    For a bandstop filter the frequencies between lower and upper are
        discarded.

    @param lower: lower pass or stop frequency limit
    @param upper: higher pass or stop frequency limit.
    @param ftype:  wether to use a bandpass or a band stop filter.

    @return A banner indicating that the filter was applied.
    """
    frequency_filter(DATA, bool(ftype), float(lower), float(upper))

    return dbc.Alert("Successfully applied bandstop filter", color="success")


@app.callback(Output("analyze-dwnsmpl-result", "children"),
              Input("analyze-dwnsmpl-apply", "n_clicks"),
              State("analyze-dwnsmpl-rate", "value"),
              prevent_initial_call=True)
def analyze_downsample(_, sampling_rate: str) -> html.Div:
    """
    Used by the preprocessing screen.

    Decimates the signal to contain as many data points as the signal would
            have if it was sampled at rate fs.
    Uses scipy.signal.decimate, which avoids aliasing.

        @param clicked: button to cause the application of the downsampling.
        @param fs: new sampling rate.

        @return a banner indicating if the downsampling was applied
    """
    downsample(DATA, int(sampling_rate))

    return dbc.Alert("Successfully downsampled", color="success")


@app.callback(Output("analyze-linenoise-result", "children"),
              Input("analyze-linenoise-apply", "n_clicks"),
              prevent_initial_call=True)
def analyze_humming(_) -> html.Div:
    """
    Used by preprocessing screen.

    Removes noise caused by the electrical system's frequency which is 50 Hz
            in Europe. I.e. removes the 50 Hz component from the signal

        @param clicked: button that causes the fitering to be applied.

        @return a banner indicating if the filter was applied
    """
    filter_el_humming(DATA)

    return dbc.Alert("Successfully removed electrical humming",
                     color="success")


# ======== Basics
@app.callback(Output("channels-table", "children", allow_duplicate=True),
              Input("analyze-snr", "n_clicks"),
              prevent_initial_call=True)
def analyze_snr(_) -> html.Div:
    """
    used by analyze screen.

    Computes the signal to noise ratio per channel and adds it to the result
        dataframe
    """
    compute_snrs(DATA)

    return generate_table(DATA.channels_df)


@app.callback(Output("channels-table", "children", allow_duplicate=True),
              Input("analyze-rms", "n_clicks"),
              prevent_initial_call=True)
def analyze_rms(_) -> html.Div:
    """
    used by analyze screen.

    Computes the root mean square/power per channel and adds it to the result
        dataframe
    """
    compute_rms(DATA)

    return generate_table(DATA.channels_df)


@app.callback(Output("channels-table", "children", allow_duplicate=True),
              Input("analyze-ent", "n_clicks"),
              prevent_initial_call=True)
def analyze_entropy(_) -> html.Div:
    """
    used by analyze screen.

    Computes the ientropy per channel and adds it to the result
        dataframe
    """
    compute_entropies(DATA)

    return generate_table(DATA.channels_df)


# @app.callback(Output("channels-table", "children", allow_duplicate=True),
#               Input("analyze-env", "n_clicks"),
#               prevent_initial_call=True)
# def analyze_envelope(clicked):
#     """
#     used by analyze screen.
#
#     Computes the envelope per channel and adds it to the result
#         dataframe
#     """
#     compute_envelopes(DATA)
#     #DATA.df['envelope'] = np.split(DATA.envelopes, DATA.envelopes.shape[0])
#
#     return generate_table(DATA.df)
#
#
# @app.callback(Output("channels-table", "children", allow_duplicate=True),
#               Input("analyze-derv", "n_clicks"),
#               prevent_initial_call=True)
# def analyze_derivative(clicked):
#     """
#     used by analyze screen.
#
#     Computes the derivative per channel and adds it to the result
#         dataframe
#     """
#     compute_derivatives(DATA)
#     #DATA.df['derivative'] = np.split(DATA.derivatives,
#                                       DATA.derivatives.shape[0])
#
#     return generate_table(DATA.df)
#
#
# @app.callback(Output("channels-table", "children", allow_duplicate=True),
#               Input("analyze-mean", "n_clicks"),
#               prevent_initial_call=True)
# def analyze_mv_average(clicked):
#     """
#     used by analyze screen.
#
#     Computes the moving average per channel and adds it to the result
#         dataframe
#     """
#     compute_mv_avg(DATA)
#     #DATA.df['mv_average'] = np.split(DATA.mv_means, DATA.mv_means.shape[0])
#
#     return generate_table(DATA.df)
#
#
# @app.callback(Output("channels-table", "children", allow_duplicate=True),
#               Input("analyze-mad", "n_clicks"),
#               prevent_initial_call=True)
# def analyze_mv_mad(clicked):
#     """
#     used by analyze screen.
#
#     Computes the moving mean absolute deviation per channel and adds it to
#       the result dataframe
#     """
#     compute_mv_mads(DATA)
#     #DATA.df['mv_mad'] = np.split(DATA.mv_mads, DATA.mv_mads.shape[0])
#
#     return generate_table(DATA.df)
#
#
# @app.callback(Output("channels-table", "children", allow_duplicate=True),
#               Input("analyze-var", "n_clicks"),
#               prevent_initial_call=True)
# def analyze_mv_var(clicked):
#     """
#     used by analyze screen.
#
#     Computes the moving variance per channel and adds it to the result
#         dataframe
#     """
#     compute_mv_vars(DATA)
#     #DATA.df['mv_var'] = np.split(DATA.mv_vars, DATA.mv_vars.shape[0])
#
#     return generate_table(DATA.df)


# ========== Spectral
# TODO use dummy, only for plotting/subsequent analysis
@app.callback(Output("channels-table", "children", allow_duplicate=True),
              Input("analyze-psd", "n_clicks"),
              prevent_initial_call=True)
def analyze_psds(_) -> html.Div:
    """
    used by analyze screen.

    Computes the power spectral densities for all selected rows and stores the
        result.
    """
    compute_psds(DATA)

    # DATA.df["psd_freq"].apply(lambda x: DATA.psds[0])
    # DATA.df['psd_power'] = np.split(DATA.psds[1], DATA.psds[1].shape[0])
    # DATA.df['psd_phase'] = np.split(DATA.psds[2], DATA.psds[2].shape[0])

    return generate_table(DATA.channels_df)


# TODO maybe store
@app.callback(Output("channels-table", "children", allow_duplicate=True),
              Input("analyze-dpsd", "n_clicks"),
              prevent_initial_call=True)
def analyze_detrend_psds(_) -> html.Div:
    """
    used by analyze screen.

    Computes the power spectral densities for all selected rows and plots the \
       #            results.
    """
    detrend_fooof(DATA)
    DATA.df['detrended_psd'] = np.split(DATA.detrended_psds,
                                        DATA.detrended_psds.shape[0])

    return generate_table(DATA.channels_df)


# Both plotting and qunatities
@app.callback(Output("channels-table", "children", allow_duplicate=True),
              Input("analyze-spec", "n_clicks"),
              prevent_initial_call=True)
def analyze_spectrograms(_) -> html.Div:
    """
    used by analyze screen.

    Computes the spectrogram for all selected rows and plots the results.
    """
    compute_spectrograms(DATA)
    DATA.df[:, 'spectrogram_freqs'] = DATA.spectrograms[0]
    DATA.df[:, 'spectrogram_time'] = DATA.spectrograms[1]
    DATA.df['spectrogram'] = DATA.spectrograms[2]

    return generate_table(DATA.channels_df)


# Quantities
@app.callback(Output("channels-table", "children", allow_duplicate=True),
              Input("analyze-aperiodic-periodic", "n_clicks"),
              prevent_initial_call=True)
def analyze_periodic_aperiodic(_) -> html.Div:
    """
    Used by analyze screen.

    Computes the PSDs for selected electrodes and then separates the periodic
                   from the aperiodic components. Stores the parameter of the
                   aperiodic component to the result dataframe
    """
    compute_periodic_aperiodic_decomp(DATA)
    DATA.df['aperiodic_offset'] = DATA.fooof_group.get_params(
                                        'aperiodic_params', 'offset')
    DATA.df['aperiodic_exponent'] = DATA.fooof_group.get_params(
                                        'aperiodic_params', 'exponent')

    return generate_table(DATA.channels_df)


# ================= TODO Activity
@app.callback(Output("channels-table", "children", allow_duplicate=True),
              Output("peaks-table", "children"),
              Input("analyze-peaks-ampl", "n_clicks"),
              State("analyze-peaks-ampl-thresh", "value"),
              prevent_initial_call=True)
def analyze_peaks(_, thresh_factor: str) -> html.Div:
    """
    used by analyze screen.

    Detects peaks in the signal by the absolute amplitude with a threshold
    based on a user defined factor (default is 3) times the mean absolute
    deviation.
    """
    if thresh_factor is not None:
        detect_peaks(DATA, float(thresh_factor))
    else:
        detect_peaks(DATA)

    return generate_table(DATA.channels_df), generate_table(DATA.peaks_df)


# @app.callback(Output("analyze-events-stats", "children"),
#               Output("analyze-events", "n_clicks"),
#               Input("analyze-events", "n_clicks"),
#               State("analyze-events-method", "value"),
#               State("analyze-events-thresh", "value"),
#               State("analyze-events-export", "value"),
#               State("analyze-events-fname", "value"),
#               prevent_initial_call=True)
# def analyze_events_moving_dev(clicked, method, thresh_factor, export, fname):
#     """
#     Used by analyze screen.

#     Detects events/bursts by computing the moving average with a large window
#        #            and a threshold.
#     """
#     export = len(export) > 0
#     res = None
#     if clicked is not None and clicked > 0:
#         if thresh_factor is None:
#             res = detect_events_moving_dev(DATA, method,
#                                            export=export, fname=fname)
#         else:
#             res = detect_events_moving_dev(DATA, method, STD_BASE_MV_STD,
#                                            float(thresh_factor),
#                                            export=export,
#                                            fname=fname)
#
#     return res, 0


# ======== Network
@app.callback(Output("network-table", "children", allow_duplicate=True),
              Input("analyze-xcorr", "n_clicks"),
              prevent_initial_call=True)
def analyze_cross_correlation(_) -> html.Div:
    """
    used by analyze screen.

    Computes the cross correlation between all selected channels and adds it to
    the results.
    """
    compute_xcorrs(DATA)
    DATA.df.loc[:, 'cross-correlation_lags'] = DATA.xcorrs[0]
    DATA.df['cross-correlation'] = DATA.xcorrs[1]

    return generate_table(DATA.network_df)


@app.callback(Output("network-table", "children", allow_duplicate=True),
              Input("analyze-mi", "n_clicks"),
              prevent_initial_call=True)
def analyze_mutual_information(_) -> html.Div:
    """
    used by analyze screen.

    Computes mutual information between all selected channels and adds them to
    the results.
    """
    compute_mutual_info(DATA)
    DATA.df['mutual_information'] = DATA.mutual_informations

    return generate_table(DATA.network_df)


@app.callback(Output("network-table", "children", allow_duplicate=True),
              Input("analyze-te", "n_clicks"),
              prevent_initial_call=True)
def analyze_transfer_entropy(_) -> html.Div:
    """
    used by analyze screen.

    Computes transfer entropy between all selected channels and adds it to te
    results.
    """
    compute_transfer_entropy(DATA)
    DATA.df['transfer_entropy'] = DATA.transfer_entropy

    return generate_table(DATA.network_df)


@app.callback(Output("network-table", "children", allow_duplicate=True),
              Input("analyze-coh", "n_clicks"),
              prevent_initial_call=True)
def analyze_coherence(_) -> html.Div:
    """
    used by analyze screen.

    Computes transfer entropy between all selected channels and adds it to the
    results.
    """
    compute_coherence(DATA)
    # TODO correct the addition to df
    DATA.df['coherence_freqs'] = DATA.coherences[:][0]
    DATA.df['coherence_lags'] = DATA.coherence[:][2]
    DATA.df['coherence'] = DATA.coherences[:][1]

    return generate_table(DATA.network_df)


@app.callback(Output("network-table", "children", allow_duplicate=True),
              Input("analyze-gc", "n_clicks"),
              prevent_initial_call=True)
def analyze_granger_causality(_) -> html.Div:
    """
    used by analyze screen.

    Computes transfer entropy between all selected channels and adds it to the
    results.
    """
    compute_granger_causality(DATA)
    DATA.df['granger_causality'] = DATA.granger_causalities

    return generate_table(DATA.network_df)


@app.callback(Output("network-table", "children", allow_duplicate=True),
              Input("analyze-sgc", "n_clicks"),
              prevent_initial_call=True)
def analyze_spectral_granger_causality(_) -> html.Div:
    """
    used by analyze screen.

    Computes transfer entropy between all selected channels and adds it to the
    results.
    """
    compute_spectral_granger(DATA)
    DATA.df['spectral_granger_freqs'] = DATA.spectral_granger[0]
    DATA.df['spectral_granger_causality'] = DATA.spectral_granger[1]

    return generate_table(DATA.network_df)


@app.callback(Output("network-table", "children", allow_duplicate=True),
              Input("analyze-csd", "n_clicks"),
              prevent_initial_call=True)
def analyze_current_source_densities(_) -> html.Div:
    """
    used by analyze screen.

    Computes transfer entropy between all selected channels and adds it to the
    results.
    """
    compute_current_source_density(DATA)
    # TODO correct the addition to df
    DATA.df['current_source_density'] = DATA.csds

    return generate_table(DATA.network_df)


# ======= Visualize
@app.callback(Output("analyze-output-dummy", "children", allow_duplicate=True),
              Input("analyze-plot-ts", "n_clicks"),
              State("analyze-plot-ts-input", "value"),
              prevent_initial_call=True)
def analyze_plot_time_series(_: int, to_plot: list[int]) -> None:
    """
    Used on select screen.

    Plots the raw signals of the selected electrodes in the given time window.

        @param t_start: the start of the selected time windw in s:ms:mus
        @param t_end: the end of the selected time window in s:ms:mus

        @return A dummy as dash callbacks require an output. The plotting is
                done in a separate process by matplotlib (as Dash plots are
                only suitable for small amounts of data)
    """
    signals = TimeSeriesPlottable.SIG.value in to_plot
    envelope = TimeSeriesPlottable.ENV.value in to_plot
    derivative = TimeSeriesPlottable.DERV.value in to_plot
    mv_average = TimeSeriesPlottable.MV_AVG.value in to_plot
    mv_mad = TimeSeriesPlottable.MV_MAD.value in to_plot
    mv_var = TimeSeriesPlottable.MV_VAR.value in to_plot
    peaks = TimeSeriesPlottable.PEAKS.value in to_plot
    bursts = TimeSeriesPlottable.BURSTS.value in to_plot
    seizure = TimeSeriesPlottable.SEIZURE.value in to_plot
    selected = True

    if DATA.envelopes is None:
        compute_envelopes(DATA)
    if DATA.derivatives is None:
        compute_derivatives(DATA)
    if DATA.mv_means is None:
        compute_mv_avgs(DATA)
    if DATA.mv_mads is None:
        compute_mv_mads(DATA)
    if DATA.channels_df.empty:
        detect_peaks(DATA)
#   if DATA.bursts is None:
#        detect_bursts(DATA)
#    if DATA.seizure is None:
#        detect_seizure(DATA)

    plot_time_series_grid(DATA, selected, signals, envelope, derivative,
                          mv_average, mv_mad, mv_var, peaks, bursts, seizure)
    return None


# ======== Export
# @app.callback(Output("analyze-animate-play", "n_clicks"),
#               Input("analyze-animate-play", "n_clicks"),
#               State("analyze-animate-fps", "value"),
#               State("analyze-animate-slow-down", "value"),
#               prevent_initial_call=True)
# def analyze_amplitude_animation(clicked: int, fps: str, slow_down: str):
#     """
#     Used by analyze screen.
#
#     Draws the electrode grid as on the MEA and color codes the current
#     absolute amplitude binned to the frame per second rate specified
#     in fps and creates a video (mp4/h264) from it of the specified
#     time window.
#
#         @param clicked: Button causing the generation of the video
#         @param fps: How many frames shall be generated per second. the lower,
#                     the wider the bins, i.e. the worse the temporal
#                     resolution of the video but the smaller the video size.
#         @param t_start: Where the video shall begin. if not specified
#                         (i.e. None), 0 is choosen.
#         @param t_stop: Where the video shall end. If not specified, the
#                        duration of the signal is chosen.
#
#         @return sets the button clicks back to 0, done because dash callbacks
#                        have to have an output
#     """
#     if clicked is not None and clicked > 0:
#         new_sr = fps / slow_down
#         bins = bin_amplitude(DATA, new_sr)
#
#     return 0


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
