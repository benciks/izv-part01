#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Simon Bencik <xbenci01> 

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any


def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    # Create array of values from a to b with steps
    x = np.linspace(a, b, steps)

    # Calculate value for each x in array
    y = (x[1:] - x[:-1]) * f((x[1:] + x[:-1]) / 2)

    # Return sum of calculated values
    return np.sum(y)

    pass


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    # Calculate function values for each a 
    range_values = np.linspace(-3, 3, 100)
    a_values = np.array(a).reshape(-1, 1)
    values = a_values**2 * range_values**3 * np.sin(range_values)

    # Transpose values for plotting
    values = values.T

    # Plot each function
    for i in range(len(a)):
        plt.plot(range_values, values[:, i], label=f"$y_{{{a[i]}}}(x)$")
        plt.fill_between(range_values, 0, values[:, i], alpha=0.1)
        plt.annotate(f"$\\int f_{{{a[i]}}}(x)dx$ = " + "{:.2f}".format(np.trapz(values[:, i], range_values)), xy=(range_values[-1], values[-1, i]))

    # Plot settings
    plt.xlabel("x")
    plt.ylabel(f"$f_{{a}}(x)$")
    plt.xlim(-3, 5)
    plt.xticks(ticks=[-3, -2, -1, 0, 1, 2, 3], labels=[-3, -2, -1, 0, 1, 2, 3])
    plt.ylim(0)
    plt.legend(bbox_to_anchor=(0.5, 1.12), loc="upper center", ncol=len(a))
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    time_range = np.linspace(0, 100, 10000)

    # Calculate values of first function
    values1 = 0.5 * np.cos(1/50 * np.pi * time_range)
    values2 = 0.25 * (np.sin(np.pi * time_range) + np.sin(3/2 * np.pi * time_range))

    # Third function is sum of first two
    values3 = values1 + values2

    fig, axs = plt.subplots(3, 1)
    plt.tight_layout()

    # Plot each function
    axs[0].plot(time_range, values1)
    axs[1].plot(time_range, values2)

    # Compute values higher than values1
    mask = values3 > values1
    values3_positive = np.where(mask, values3, values1)
    values3_negative = np.where(~mask, values3, values1)

    # Set the rest of values to nan so they are not plotted
    values3_positive[~mask] = np.nan
    values3_negative[mask] = np.nan

    # Plot third function
    axs[2].plot(time_range, values3_positive, color="g")
    axs[2].plot(time_range, values3_negative, color="r")

    for i in range(3):
        axs[i].set(ylabel=f"$f_{{{i}}}(t)$", xlabel="t")
        axs[i].set_xlim(0, 100)
        axs[i].set_ylim(-0.8, 0.8)
        axs[i].xaxis.set_ticks(np.arange(0, 101, 20))
        axs[i].yaxis.set_ticks(np.arange(-0.8, 0.9, 0.4))

    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()

def download_data() -> List[Dict[str, Any]]:
    r = requests.get("https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html")
    soup = BeautifulSoup(r.content, "html.parser")

    # Get all table rows with class "nezvyraznit"
    rows = soup.find_all("tr", class_="nezvyraznit")

    output = []

    # Iterate over rows
    for row in rows:
        # Get all cells
        cells = row.find_all("td")

        position = cells[0].text.strip()

        # Remove degree symbol from latitude and longitude, replace comma with dot
        lat = cells[2].text.strip().replace("°", "").replace(",", ".")
        long = cells[4].text.strip().replace("°", "").replace(",", ".")

        # Replace comma with dot  
        height = cells[6].text.strip().replace(",", ".")

        row = {
            "position": position,
            "lat": float(lat),
            "long": float(long),
            "height": float(height)
        }

        output.append(row)

    return output

