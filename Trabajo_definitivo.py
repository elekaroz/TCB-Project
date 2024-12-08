"""
Cluster Membership
Authors: P. González-Berdayes, E. Lekaroz-Urriza, J. Prieto-Polo & E. Urquijo Rodríguez
"""

# We import the packages needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia

# We define a class called "cluster_membership" in order to be used for every cluster
class cluster_membership:
    # We provide as arguments the name of the cluster and the parameters needed
    # for carrying on the cone-search in Gaia
    def __init__(self, name, ra, dec, radius):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.radius = radius
    # We define a function that performs the cone-search imposing a maximum error
    # in the parallaxes and the fluxes, as well as a maximum magnitud of 18
    def cone_search(self):
        gaia_query = Gaia.launch_job_async(f"""SELECT source_id, ra, ra_error, dec,\
                                 dec_error, parallax, parallax_error,\
                                 pmra, pmra_error, pmdec, pmdec_error,\
                                 radial_velocity, radial_velocity_error,\
                                 phot_g_mean_mag, phot_bp_mean_mag,\
                                 phot_rp_mean_mag,bp_rp,\
                                 astrometric_excess_noise,\
                                 phot_bp_rp_excess_factor,\
                                 DISTANCE({ra}, {dec}, ra, dec)\
                                 FROM gaiadr3.gaia_source\
                                 WHERE DISTANCE({ra}, {dec}, ra, dec) < {radius} / 60. \
                                 AND parallax_over_error > 10\
                                 AND phot_bp_mean_flux_over_error > 10\
                                 AND phot_rp_mean_flux_over_error > 10\
                                 AND phot_g_mean_mag < 18""",
                                 dump_to_file=False)
        # To avoid saving the data in the server we use async
        # We filter stars with errors too large, or apparent magnitud too small
        # Units in degrees
        # An f-string allows to embed expressions inside string literals, within {}
        gaia_data = gaia_query.get_results()
        # Now we convert the results in a data frame
        df = gaia_data.to_pandas()
        parallax = df["parallax"]
        ra_pm = df["pmra"]
        dec_pm = df["pmdec"]
        g_mag = df["phot_g_mean_mag"]
        bp_rp = df["bp_rp"]
        r_vel = df["radial_velocity"]
        # We get back lists for the data of these parameters for every star, plus the data frame
        return df, parallax, ra_pm, dec_pm, g_mag, bp_rp, r_vel 
        
    # Now it is turn of defining another funcion that gives us the indexes of each
    # magnitud that has been extracted
    def series_str(self):
        series_str = pd.Series(["parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "radial_velocity", "radial_velocity_error", "phot_g_mean_mag", "bp_rp", "astrometric_excess_noise", "phot_bp_rp_excess_factor"])
        print("""The parameters for which the clusters can be filtered are:""")
        print(series_str)
        # We need a list with the indexes for every available parameter, so we can call them back later
        return series_str
    
    # We define a function that selects the data according to a filter imposed 
    # by ourselves.
    def filtering (self, data_frame, series_str, ind, mini, maxi):
        # data_frame: the data you want to filter
        # series_str: the set of parameters in which the one you want to filter by is included
        # ind: the index of the parameter you want to filter by
        # mini: the minimum value that takes the parameter
        # maxi: the maximum value that takes the parameter
        
        # Now we filter the data of the defined parameter between the defined limits
        fltd_min = data_frame[data_frame[series_str[ind]] < maxi]
        fltd = fltd_min[mini < fltd_min[series_str[ind]]] 
        # We return the reduced data frame
        return fltd
    
    # We define a function that plots in 2D the data associated to 2 of the parameters
    # extracted from Gaia. It is possible to plot as well the reduced data frame 
    # with the stars that belong to a certain cluster
    def plot_set (self, data_frame, series_str, indx, indy, reduced_data_frame = None, xtext = 'x data', ytext = 'y data', xlimit = None, ylimit = None, savefig=False):
        # data_frame: the data you want to plot and filter
        # series_str: the set of parameters in which the one you want to filter by is included
        # indx: the index of the parameter that is going to be plotted in the x axis
        # indy: the index of the parameter that is going to be plotted in the y axis
        # reduced_data_frame: the reduced data you have already filter (only if you want to plot it)
        # xtext & ytext: the titles in the x and y axis
        # xlimit & ylimit: the limits in the x and y axis (optional)
        # savefig: it is possible to save the figure by selecting "True"

        # We define the x and y data to be plotted
        x_data = data_frame[series_str[indx]]
        y_data = data_frame[series_str[indy]]
        
        # We clean for nans if they are present in the sample
        x_data = [i for i in x_data if np.isnan(i)==False]
        y_data = [j for j in y_data if np.isnan(j)==False]

        # We define the figure, in order to be included some histograms as well 
        fig = plt.figure(figsize = (8, 6))
        gs = fig.add_gridspec(4, 4)  # Create a grid of subplots

        ax_main = fig.add_subplot(gs[1:4, 0:3]) # Place of the main plot in the grid of subplots

        # We plot the data
        ax_main.scatter(x_data, y_data, s = 7, c = 'darkorange', label = 'All data')

        ax_hist_x = fig.add_subplot(gs[0, 0:3], sharex=ax_main) # Place of both histograms in the grid
        ax_hist_y = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

        ax_hist_x.get_xaxis().set_visible(False) # Hide the axes
        ax_hist_y.get_yaxis().set_visible(False)
        
        bin_width_x = (max(x_data) - min(x_data)) / 50  # We divide the sample into 50 bins
        bin_width_y = (max(y_data) - min(y_data)) / 50
        
        bins_x = np.arange(min(x_data), max(x_data) + bin_width_x, bin_width_x) # We define the bins
        bins_y = np.arange(min(y_data), max(y_data) + bin_width_y, bin_width_y)

        ax_hist_x.hist(x_data, bins = bins_x, color='darkorange', alpha=0.5) # We plot the histograms
        ax_hist_y.hist(y_data, bins = bins_y, color='darkorange', alpha=0.5, orientation='horizontal')

        # If we have selected a reduced sample of data in order to be plotted
        # as well, we plot it
        if reduced_data_frame is not None:
            
            x_data_reduced = reduced_data_frame[series_str[indx]] # Define the x and y data
            y_data_reduced = reduced_data_frame[series_str[indy]]

            ax_main.scatter(x_data_reduced, y_data_reduced, s = 7, c = 'purple', label = str(name)) # We plot them

            bins_x = np.arange(min(x_data_reduced), max(x_data_reduced) + bin_width_x, bin_width_x) # We define the bins
            bins_y = np.arange(min(y_data_reduced), max(y_data_reduced) + bin_width_y, bin_width_y)

            ax_hist_x.hist(x_data_reduced, bins= bins_x, color='purple', alpha=0.5) # And place the histograms
            ax_hist_y.hist(y_data_reduced, bins= bins_y, color='purple', alpha=0.5, orientation='horizontal')

        ax_main.tick_params(axis='both', labelsize=10)  # Main plot tick labels
        ax_hist_x.tick_params(axis='y', labelsize=10)  # Histograms tick labels
        ax_hist_y.tick_params(axis='x', labelsize=10)

        # Define the title and place the legend
        fig.suptitle(f'Data filtering: {name}', fontsize = 18)
        ax_main.legend(loc='upper right', fontsize=10)
        fig.subplots_adjust(hspace=0.3,wspace=0.3)
        # And place the labels in the axis
        ax_main.set_xlabel (xtext, fontsize = 12)
        ax_main.set_ylabel (ytext, fontsize = 12)

        # If we require a limit in the x axis we put it
        if xlimit is not None:
            ax_main.set_xlim(xlimit[0], xlimit[1])
        # And the same with the y axis
        if ylimit is not None:
            ax_main.set_ylim(ylimit[0], ylimit[1])

        # Save the figure (as pdf) if required
        if savefig == True:
            file_name = input ('Name of the figure: ')
            plt.savefig('./'+str(file_name)+'.pdf')

        # Show the figure
        fig.tight_layout()
        plt.show()

        # Now we print the number of stars that belong to each set of data
        if reduced_data_frame is None:
            print('The number of stars in the set containing all data is '+str(len(data_frame))+'.')
        if reduced_data_frame is not None:
            print('The number of stars in the set containing all data is '+str(len(data_frame))+'.')
            print('The number of stars that belong to '+str(name)+' is '+str(len(reduced_data_frame))+'.')
            
    # We define a function that computes the absolute magnitude of every star, given its apparent magnitude
    # It is possible to do it by giving an input for the value of the parallax
    def g_abs(self, data, fix_parallax = False):
        # data: the data containing the phot_g_mean_mag and parallax in order to compute the absolute magnitude
        # fix_parallax: it is possible to introduce the parallax manually for no to depend on the value given by Gaia
        
        data=data.copy() # We make a copy of the data in order to work with
        # We perform the calculation in two ways so as if you want to input the parallax or not
        if fix_parallax == True:
            # We ask for the parallax and compute de absolute magnitude
            parallax = float(input('Introduce the parallax of the stars in order to compute the absolute magnitude: '))
            data.loc[:,"g_abs"] = data["phot_g_mean_mag"] + 5*np.log10(parallax/100)
            
        else:
            # If we do not want to put it manually, it is done with the value extracted by Gaia
            data.loc[:,"g_abs"] = data["phot_g_mean_mag"] + 5*np.log10(data["parallax"]/100)
        
        return data
            
    # We define a function that plots the color magnitud diagram of a set of stars
    def plot_cm_diagram(self, reduced_data_frame, rest_data = None, savefig = False, xlimit = None, ylimit = None, histograms = False):
        # reduced_data_frame: the data you want to plot (it is supposed to have been filtered previously)
        # rest_data: the rest of the data extracted in the query that do not belong to the cluster (optional)
        # savefig: it is possible to save the figure by selecting "True"
        # xlimit & ylimit: the limits in the x and y axis (optional)
        # histograms: it is possible to plot histograms in orden to see the distribution of the stars along the CMD

        # We divide the function into two parts whether you want to plot an histogram or not
        if histograms == True:
            # Now we divide it again whether you want to plot the rest of the stars that do not belong to the cluster
            if rest_data is not None:
                
                y_data = rest_data["g_abs"] # Define the magnitudes that are going to be plotted
                x_data = rest_data["bp_rp"]
                
                fig = plt.figure(figsize = (6,8)) # Define the figure
                gs = fig.add_gridspec(4, 4)

                ax_main = fig.add_subplot(gs[1:4, 0:3])

                ax_main.scatter(x_data, y_data, s = 7, c = 'darkorange', label = 'Rest of the stars') # Plot the rest of the stars

                ax_hist_x = fig.add_subplot(gs[0, 0:3], sharex=ax_main) # Place the histograms
                ax_hist_y = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

                ax_hist_x.get_xaxis().set_visible(False) # Hide the axis
                ax_hist_y.get_yaxis().set_visible(False)

                bin_width_x = (max(x_data) - min(x_data)) / 30  # Divide the sample into 30 bins
                bin_width_y = (max(y_data) - min(y_data)) / 30

                bins_x = np.arange(min(x_data), max(x_data) + bin_width_x, bin_width_x) # Define the bins
                bins_y = np.arange(min(y_data), max(y_data) + bin_width_y, bin_width_y)

                ax_hist_x.hist(x_data, bins = bins_x, color='darkorange', alpha = 0.5) # Plot the histograms
                ax_hist_y.hist(y_data, bins = bins_y, color='darkorange', alpha = 0.5, orientation='horizontal')
                
                y_data_reduced = reduced_data_frame["g_abs"] # Define the magnitudes that are going to be plotted
                x_data_reduced = reduced_data_frame["bp_rp"]

                ax_main.scatter(x_data_reduced, y_data_reduced, s = 7, c = 'purple', label = str(name)) # Plot the stars that really belong to the cluster

                bins_x = np.arange(min(x_data_reduced), max(x_data_reduced) + bin_width_x, bin_width_x) # Define the new bins
                bins_y = np.arange(min(y_data_reduced), max(y_data_reduced) + bin_width_y, bin_width_y)

                ax_hist_x.hist(x_data_reduced, bins = bins_x , color='purple', alpha=0.5) # And place them
                ax_hist_y.hist(y_data_reduced, bins = bins_y , color='purple', alpha=0.5, orientation='horizontal')
              
            # If there is no set of data associated to the rest of the stars
            # we only plot the stars that have cluster membership
            else:
                
                y_data = reduced_data_frame["g_abs"] # Define the magnitudes that are going to be plotted
                x_data = reduced_data_frame["bp_rp"]

                fig = plt.figure(figsize = (6,8)) # Define the figure
                gs = fig.add_gridspec(4, 4)

                ax_main = fig.add_subplot(gs[1:4, 0:3])

                ax_main.scatter(x_data, y_data, s = 7, c = 'purple', label = str(name)) # Plot the stars in the CMD

                ax_hist_x = fig.add_subplot(gs[0, 0:3], sharex=ax_main) # Place the histograms
                ax_hist_y = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

                ax_hist_x.get_xaxis().set_visible(False) # Hide the axis
                ax_hist_y.get_yaxis().set_visible(False)

                bin_width_x = (max(x_data) - min(x_data)) / 30  # Divide the sample into 30 bins
                bin_width_y = (max(y_data) - min(y_data)) / 30

                bins_x = np.arange(min(x_data), max(x_data) + bin_width_x, bin_width_x) # Define the bins
                bins_y = np.arange(min(y_data), max(y_data) + bin_width_y, bin_width_y)

                ax_hist_x.hist(x_data, bins = bins_x, color='purple', alpha = 0.5) # Plot the histograms
                ax_hist_y.hist(y_data, bins = bins_y, color='purple', alpha = 0.5, orientation='horizontal')
                
            ax_main.tick_params(axis='both', labelsize=10)  # Main plot tick labels
            ax_hist_x.tick_params(axis='x', labelsize=10)  # Histograms tick labels
            ax_hist_y.tick_params(axis='y', labelsize=10)

            ax_main.invert_yaxis() # Invert the y axis, as it is a CMD
            # Define the title and place the legend
            fig.suptitle(f'Color-Magnitude Diagram: {name}', fontsize = 18)
            plt.legend(loc='upper right', fontsize=10)
            fig.subplots_adjust(hspace=0.3,wspace=0.3)
            # And place the labels in the axis
            ax_main.set_xlabel (r'$G_{BP}-G_{RP}$ [mag]', fontsize = 12)
            ax_main.set_ylabel (r'$M_G$ [mag]', fontsize = 12)
           
            # If we require a limit in the x axis we put it
            if xlimit is not None:
                ax_main.set_xlim(xlimit[0], xlimit[1])
            # And the same with the y axis
            if ylimit is not None:
                ax_main.set_ylim(ylimit[0], ylimit[1])

            # Save the figure (as pdf) if required
            if savefig == True:
                file_name = input ('Name of the figure: ')
                plt.savefig('./'+str(file_name)+'.pdf')
                
            # Show the figure
            fig.tight_layout()
            plt.show()

            # Now we print the number of stars that belong to each set of data
            if rest_data is None:
                print('The number of stars in the diagram is '+str(len(reduced_data_frame))+'.')
            if rest_data is not None:
                print('The total number of stars in the diagram is '+str(len(rest_data)+len(reduced_data_frame))+'.')
                print('The number of stars in the diagram that belong to '+str(name)+' is '+str(len(reduced_data_frame))+'.')

        # Now we consider the case in which we do not want the histograms to be plotted
        if histograms == False:
            # And divide the path whether we want to plot the stars that do not belong to the cluster
            if rest_data is not None:
                
                y_data = rest_data["g_abs"] # Define the magnitudes that are going to be plotted
                x_data = rest_data["bp_rp"]
                
                fig = plt.figure(figsize = (6,8)) # Define the figure
                plt.scatter(x_data, y_data, s = 7, c = 'darkorange', label = 'Rest of the stars') # Plot the rest of the stars

                y_data_reduced = reduced_data_frame["g_abs"] # Define the magnitudes of the stars that belong to the cluster
                x_data_reduced = reduced_data_frame["bp_rp"]

                plt.scatter(x_data_reduced, y_data_reduced, s = 7, c = 'purple', label = str(name)) # Plot the stars that really belong to the cluster
            
            # If there is no set of data associated to the rest of the stars, we only plot the stars that have cluster membership
            else:
                
                y_data = reduced_data_frame["g_abs"] # Define the magnitudes that are going to be plotted
                x_data = reduced_data_frame["bp_rp"]

                fig = plt.figure(figsize = (6,8)) # Define the figure
                plt.scatter(x_data, y_data, s = 7, c = 'purple', label = str(name)) # Plot the stars in the CMD

            plt.tick_params(axis='both', labelsize=10)  # Main plot tick labels

            plt.gca().invert_yaxis() # Invert the y axis, as it is a CMD
            # Define the title and place the legend
            fig.suptitle(f'Color-Magnitude Diagram: {name}', fontsize = 18)
            plt.legend(loc='upper right', fontsize=10)
            fig.subplots_adjust(hspace=0.3,wspace=0.3)
            # And place the labels in the axis
            plt.xlabel (r'$G_{BP}-G_{RP}$ [mag]', fontsize = 12)
            plt.ylabel (r'$M_G$ [mag]', fontsize = 12)

            # If we require a limit in the x axis we put it
            if xlimit is not None:
                plt.xlim(xlimit[0], xlimit[1])
            # And the same with the y axis
            if ylimit is not None:
                plt.ylim(ylimit[0], ylimit[1])

            # Save the figure (as pdf) if required
            if savefig == True:
                file_name = input ('Name of the figure: ')
                plt.savefig('./'+str(file_name)+'.pdf')

            # Show the figure
            fig.tight_layout()
            plt.show()
            
            # Now we print the number of stars that belong to each set of data
            if rest_data is None:
                print('The number of stars in the diagram is '+str(len(reduced_data_frame))+'.')
            if rest_data is not None:
                print('The total number of stars in the diagram is '+str(len(rest_data)+len(reduced_data_frame))+'.')
                print('The number of stars in the diagram that belong to '+str(name)+' is '+str(len(reduced_data_frame))+'.')

    # Finally we define an extra function that helps the user and explains how each function of the program works
    def help(self):
        print("""
        cone_search: performs the cone search query for the specified coordinates and provides data frame with all the parameters, parallax, proper motion in right ascension and declination, magnitud in g band, color BP-RP and radial velocity.

        series_str: provides a list with the indexes of every parameter, so it can be used in the analysis functions and the data filtering.

        filtering: filters the data for the parameter specified by the index and for the specified limits.

        plot_set: provides a figure comparing the values of two different parameters specified by the index for every star, for one or two sets.

        g_abs: computes the absolute magnitude of each star in the sample given as an argument. The parallax can be either taken by Gaia or asked to be input manually

        plot_cm_diagram: provides the color-magnitud diagram of the group of stars specified, for one or two sets of stars.

        """)
    
###############################################################################        
# Now we do the cluster membership by filtering the data according to the plots
###############################################################################

# OPEN CLUSTERS
# First we define de parameters that caracterize the cluster
name = 'M6'
ra = 265.0691667 # degrees
dec = -32.2419444 # degrees
radius = 20.00/2 # arcmin
# Now we call the class
M6 = cluster_membership(name, ra, dec, radius)
# And we do the cone-search
data_frame = M6.cone_search()[0]
# We may want to print the help
M6.help()
# Define the indexes
indexes = M6.series_str()
# Filter the data and plot the results
data_filt_par = M6.filtering(data_frame, indexes, 0, 2, 2.5)
M6.plot_set(data_frame, indexes, 0, 1, data_filt_par, xtext = r'$\text{Parallax}$ [mas]', ytext = r'$\varepsilon_{\text{Parallax}}$ [mas]')
# Define a set with the stars of the first sample that do not belong to the cluster
rest_data = data_frame.loc[~data_frame.index.isin(data_filt_par.index)]
# Compute the absolute magnitude in order to plot the CMD
data_filt_par = M6.g_abs(data_filt_par)
rest_data = M6.g_abs(rest_data)
# Plot the CMD
M6.plot_cm_diagram(data_filt_par, rest_data)
# THE PROCEEDING IS EXACTLY THE SAME WITH THE REST OF THE CLUSTERS


name = 'M45'
ra = 56.6008333 # degrees
dec = 24.1138889 # degrees
radius = 76.86/2 # arcmin

M45 = cluster_membership(name, ra, dec, radius)

data_frame = M45.cone_search()[0]

indexes = M45.series_str()

data_filt_par = M45.filtering(data_frame, indexes, 0, 6.5, 8.5)
M45.plot_set(data_frame, indexes, 0, 1, data_filt_par, xtext = r'$\text{Parallax}$ [mas]', ytext = r'$\varepsilon_{\text{Parallax}}$ [mas]')

rest_data = data_frame.loc[~data_frame.index.isin(data_filt_par.index)]

data_filt_par = M45.g_abs(data_filt_par)
rest_data = M45.g_abs(rest_data)

M45.plot_cm_diagram(data_filt_par, rest_data)


name = 'M50'
ra = 105.6841667 # degrees
dec = -8.3650000 # degrees
radius = 31.6/2 # arcmin

M50 = cluster_membership(name, ra, dec, radius)

data_frame = M50.cone_search()[0]

indexes = M50.series_str()

data_filt_par = M50.filtering(data_frame, indexes, 0, 0.9, 1.1)
M50.plot_set(data_frame, indexes, 0, 1, data_filt_par, xtext = r'$\text{Parallax}$ [mas]', ytext = r'$\varepsilon_{\text{Parallax}}$ [mas]')
data_filt_par = M50.filtering(data_filt_par, indexes, 2, -1.8, 0.8)
data_filt_par = M50.filtering(data_filt_par, indexes, 4, -1.8, 0.5)
M50.plot_set(data_frame, indexes, 2, 4, data_filt_par, xtext = r'$PM_{RA}$ [mas/yr]', ytext = r'$PM_{DEC}$ [mas/yr]', xlimit = [-10,10], ylimit = [-10,10])
M50.plot_set(data_frame, indexes, 2, 4, data_filt_par, xtext = r'$PM_{RA}$ [mas/yr]', ytext = r'$PM_{DEC}$ [mas/yr]')

rest_data = data_frame.loc[~data_frame.index.isin(data_filt_par.index)]

data_filt_par = M50.g_abs(data_filt_par)
rest_data = M50.g_abs(rest_data)

M50.plot_cm_diagram(data_filt_par, rest_data)


# GLOBULAR CLUSTERS
name = 'M4'
ra = 245.8967500 # degrees
dec = -26.5257500 # degrees
radius = 26.3/2 # arcmin 

M4 = cluster_membership(name, ra, dec, radius)

data_frame = M4.cone_search()[0]

indexes = M4.series_str()

data_filt_par = M4.filtering(data_frame, indexes, 2, -16, -9)
data_filt_par = M4.filtering(data_filt_par, indexes, 4, -23, -16)
M4.plot_set(data_frame, indexes, 2, 4, data_filt_par, xtext = r'$PM_{RA}$ [mas/yr]', ytext = r'$PM_{DEC}$ [mas/yr]')

rest_data = data_frame.loc[~data_frame.index.isin(data_filt_par.index)]

data_filt_par = M4.g_abs(data_filt_par, fix_parallax = True)
rest_data = M4.g_abs(rest_data)

M4.plot_cm_diagram(data_filt_par, rest_data)


name = 'M13'
ra = 250.4234750 # degrees
dec = 36.4613194 # degrees
radius = 120/2 # arcmin 

M13 = cluster_membership(name, ra, dec, radius)

data_frame = M13.cone_search()[0]

indexes = M13.series_str()

data_filt_par = M13.filtering(data_frame, indexes, 2, -9, 7)
data_filt_par = M13.filtering(data_filt_par, indexes, 4, -16, 8)
M13.plot_set(data_frame, indexes, 2, 4, data_filt_par, xtext = r'$PM_{RA}$ [mas/yr]', ytext = r'$PM_{DEC}$ [mas/yr]', xlimit=[-30,30], ylimit=[-30,30])
M13.plot_set(data_frame, indexes, 2, 4, data_filt_par, xtext = r'$PM_{RA}$ [mas/yr]', ytext = r'$PM_{DEC}$ [mas/yr]')

rest_data = data_frame.loc[~data_frame.index.isin(data_filt_par.index)]

data_filt_par = M13.g_abs(data_filt_par)
rest_data = M13.g_abs(rest_data)

M13.plot_cm_diagram(data_filt_par, rest_data)


name = 'M15'
ra = 322.4930417 # degrees
dec = 12.1670000 # degrees
radius = 120/2

M15 = cluster_membership(name, ra, dec, radius)

data_frame = M15.cone_search()[0]

indexes = M15.series_str()

data_filt_par = M15.filtering(data_frame, indexes, 2, -12, 10)
data_filt_par = M15.filtering(data_filt_par, indexes, 4, -16, 8)
M15.plot_set(data_frame, indexes, 2, 4, data_filt_par, xtext = r'$PM_{RA}$ [mas/yr]', ytext = r'$PM_{DEC}$ [mas/yr]', xlimit=[-40,40], ylimit=[-50,30])
M15.plot_set(data_frame, indexes, 2, 4, data_filt_par, xtext = r'$PM_{RA}$ [mas/yr]', ytext = r'$PM_{DEC}$ [mas/yr]')

rest_data = data_frame.loc[~data_frame.index.isin(data_filt_par.index)]

data_filt_par = M15.g_abs(data_filt_par)
rest_data = M15.g_abs(rest_data)

M15.plot_cm_diagram(data_filt_par, rest_data)


