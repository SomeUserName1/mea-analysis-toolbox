"""
Dash-based HTML code to show event/burst statistics in the browser via the Dash server.
"""
import time
import csv

from dash import html
import dash_bootstrap_components as dbc


def show_events(data, selected_names):
    children = []
    
    header = html.Thead(html.Tr([html.Th("Name"), html.Th("Duration"), html.Th("Start"), html.Th("Stop"), html.Th("Spike Count"), html.Th("Spike Rate"), html.Th("RMS"), html.Th("Max. Amplitude"), html.Th("Mean ISI"), html.Th("Delta"), html.Th("Theta"), html.Th("Alpha"), html.Th("Beta"), html.Th("Gamma"), html.Th("Delay")]))

    for event in data.events:
       children.append(
           html.Tr([
               html.Td(str(selected_names[data.selected_rows.index(event.electrode_idx)])),
               html.Td(str(event.duration)),
               html.Td(str(event.start_idx * 1 / data.sampling_rate)),
               html.Td(str(event.end_idx * 1 / data.sampling_rate)),
               html.Td(str(event.spike_count)),
               html.Td(str(event.spike_rate)),
               html.Td(str(event.rms)),
               html.Td(str(event.max_amplitude)),
               html.Td(str(event.mean_isi)),
               html.Td(event.band_powers['delta']),
               html.Td(event.band_powers['theta']),
               html.Td(event.band_powers['alpha']),
               html.Td(event.band_powers['beta']),
               html.Td(event.band_powers['gamma']),
               html.Td(str(event.delay)),
                   ])
           )

    body = html.Tbody(children)

    return dbc.Table([header, body], bordered=True)



def export_events(data, selected_names, fname=None):
    fields = ["Name", "Duration", "Start", "Stop", "Spike Count", "Spike Rate", "RMS", "Max. Amplitude", "Mean ISI", "Delta", "Theta", "Alpha", "Beta", "Gamma", "Delay"]
    
    if fname is None:
        fname = "events_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"     
    with open(fname, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)

        for event in data.events:
           csv_writer.writerow(
               [
                    str(selected_names[data.selected_rows.index(event.electrode_idx)]),
                    str(event.duration),
                    str(event.start_idx * 1 / data.sampling_rate),
                    str(event.end_idx * 1 / data.sampling_rate),
                    str(event.spike_count),
                    str(event.spike_rate),
                    str(event.rms),
                    str(event.max_amplitude),
                    str(event.mean_isi),
                    str(event.band_powers['delta']),
                    str(event.band_powers['theta']),
                    str(event.band_powers['alpha']),
                    str(event.band_powers['beta']),
                    str(event.band_powers['gamma']),
                    str(event.delay)
                ]
           )
