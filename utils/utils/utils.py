import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import skew
import matplotlib.pyplot as plt
from utils.constants.constants import conversion_factor_for_densities
import scipy as sc

def get_active_drilling_data(
    df: pd.DataFrame, sim_df: Optional[pd.DataFrame] = None, include_rop_zeros=False
):
    """
    Returns the active drilling data from the given DataFrame. If `sim_df` is provided,
    it also returns the simulated drilling data corresponding to the active drilling data.

    Returns:
        pd.DataFrame: The DataFrame containing the active drilling data.
        pd.DataFrame, optional: The DataFrame containing the simulated drilling data
        corresponding to the active drilling data. Only returned if `sim_df` is not None.
    Note:
        Funcitonality might be limited if data consists of several seperate drilling
        segments.
    """

    if include_rop_zeros:
        rop_cum = df["ROP"].cumsum()
        drilling_df = df.loc[(rop_cum > 0) & (rop_cum < rop_cum.max())]
    else:
        drilling_df = df[df["ROP"] > 0]

    if sim_df is None:
        return drilling_df

    sim_drilling_df = sim_df.loc[drilling_df.index]
    return drilling_df, sim_drilling_df


def reading_labeled_drill_report_data(FILEPATH):
    """
    This function reads the labeled drill_report_data from a .txt file
    The format of the txt file is timestamp_start/timestamp_end/label
    The delimiter is the / sign


    Returns: A list containing time_stamp_start, time_stamp_end and the label as
    values for each element.
    """
    labeled_sections = []

    with open(FILEPATH, "r") as file:
        lines = file.readlines()
        for line in lines:
            split = line.split("/")
            split[-1] = int(split[-1])
            labeled_sections.append(split)

    return np.array(labeled_sections)
def extracting_labeled_data_from_dataframe(df : pd.DataFrame, labeled_sections : np.array):
    """
    Iterates through the labeled sections for the drilling operation and makes a new
    dataframe with these values.

    Returns: a pandas dataframe with the labeled drilling data
    """
    dictionary_of_sections = {}
    n = 0
    for section in labeled_sections:
        n+=1
        temporary_dataframe = df.loc[section[0]:section[1]].copy()
        temporary_dataframe["labels"] =section[-1].astype(np.int32)
        dictionary_of_sections[f'Section {n}'] = temporary_dataframe
    return dictionary_of_sections

def plot_sections(dictionary_of_sections, variable):
    """
    plots the relevant dimensionless numbers for the dictionary returned in the function
    above.
    """
    for section in dictionary_of_sections:
        fig,ax = plt.subplots(1, figsize = (20,10))
        ax.set_title(f"{variable} for {section}")
        ax.plot(dictionary_of_sections[section][variable].values, color = "C0")
        # ax[1].set_title(f"Taylor number for {section}")
        # ax[1].plot(dictionary_of_sections[section]["TaylAnn"].values, color = "C1")
        # ax[2].set_title(f"Rouse number for {section}")
        # ax[2].plot(dictionary_of_sections[section]["RouseAnn"].values,color = "C2")
        # ax[3].set_title(f"Shield number for {section}")
        # ax[3].plot(dictionary_of_sections[section]["ShieldAnn"].values,color = "C3")
        # ax[4].set_title(f"Reynolds vs Shield number for {section}")
        # ax[4].scatter(dictionary_of_sections[section]["ReynAnn"].values,dictionary_of_sections[section]["ShieldAnn"].values,color = "C0")
        # ax[5].set_title(f"Taylor vs Shield number for {section}")
        # ax[5].scatter(dictionary_of_sections[section]["TaylAnn"].values,dictionary_of_sections[section]["ShieldAnn"].values,color = "C1")
        # ax[6].set_title(f"Rouse vs Shield number for {section}")
        # ax[6].scatter(dictionary_of_sections[section]["RouseAnn"].values,dictionary_of_sections[section]["ShieldAnn"].values,color = "C2")
        # plt.suptitle(f"Dimensionless numbers for annulus Label: {dictionary_of_sections[section]["labels"][0]}")
        plt.tight_layout()
def change_directory():
    import os
    current_directory = os.getcwd()
    while current_directory.endswith("Notebooks"):
      os.chdir("..")
      current_directory = os.getcwd()
      print("Current working directory: ", current_directory)




        


                
def plot_with_warnings_down_hole_ecd(df, warnings):
    x = np.arange(0,len(df),1)
    warnings_indices = [element[1] for element in warnings]
    fig, ax = plt.subplots(5,figsize = (20,20))
    ax[0].plot(df["DH_PRESS_ECD"].values, label = "Downhole ECD")
    ax[0].scatter(x[warnings_indices],df["DH_PRESS_ECD"].iloc[warnings_indices], color = "C3")
    ax[0].legend()
    ax[1].plot(df["TORQ"].values, label = "Torque")
    ax[1].scatter(x[warnings_indices],df["TORQ"].iloc[warnings_indices], color = "C3")
    ax[1].legend()
    ax[2].plot(df["MUD_FLOW_IN"].values, label = "Flow in")
    ax[2].scatter(x[warnings_indices],df["MUD_FLOW_IN"].iloc[warnings_indices], color = "C3")
    ax[2].legend()
    ax[3].plot(df["RPM_SURF"].values, label = "rpm")
    ax[3].scatter(x[warnings_indices],df["RPM_SURF"].iloc[warnings_indices], color = "C3")
    ax[3].legend()
    ax[4].plot(df["ROP"].values, label = "ROP")
    ax[4].scatter(x[warnings_indices],df["ROP"].iloc[warnings_indices], color = "C3")
    ax[4].legend()


def plot_with_warnings_pack_off_sensor_1_2(df_input, df_sim, warnings, chunk_size):
    x = np.arange(0,len(df_input),1)
    warnings_indices = [element[1] for element in warnings]
    df_input = df_input.reset_index(drop = True)
    df_sim = df_sim.reset_index(drop = True)
    df_split = [df_input.iloc[i:i+chunk_size] for i in range(0,len(df_input),chunk_size)]
    df_sim_split = [df_sim.iloc[i:i+chunk_size] for i in range(0,len(df_sim),chunk_size)]
 
    num_segments = len(df_split)
    warnings_per_chunk = [[] for _ in range(num_segments)]
    n = 1
    for warning_index in warnings_indices:

        for i, df in enumerate(df_split):
            if df.index[0] <= warning_index <= df.index[-1]:
                warnings_per_chunk[i].append(warning_index)
                break

    
    
    
    for df,df_s, warning in zip(df_split,df_sim_split,warnings_per_chunk):
        fig, ax = plt.subplots(8,figsize = (20,20))
        x_vals_for_plot = np.arange(df.index[0], df.index[-1]+1,1)
        ax[0].plot(x_vals_for_plot,df["ASMECD1-T"].values, label = "ASM 1")
        ax[0].plot(x_vals_for_plot,df["ASMECD2-T"].values, label = "ASM 2")
        ax[0].plot(x_vals_for_plot,df_s["ecdAtPos2"].values, label = "HFM 1")
        ax[0].plot(x_vals_for_plot,df_s["ecdAtPos3"].values, label = "HFM 2")
        ax[0].scatter(warning,df["ASMECD2-T"].loc[warning], color = "C3")
        ax[0].set_ylabel("ECD [$\mathrm{gcm}^{-3}$]")
        ax[0].legend()
        torque = df["TORQ"].values
        ax[1].plot(x_vals_for_plot,torque, label = "Torque")
        ax[1].scatter(warning,df["TORQ"].loc[warning], color = "C3")
        ax[1].set_ylabel("[Nm]")
        ax[1].legend()
        ax[2].plot(x_vals_for_plot,df["MUD_FLOW_IN"].values, label = "Flow in")
        ax[2].scatter(warning,df["MUD_FLOW_IN"].loc[warning], color = "C3")
        ax[3].set_ylabel("[$\mathrm{ft}^3 / \mathrm{s}$]")
        ax[2].legend()
        ax[3].plot(x_vals_for_plot,df["RPM_SURF"].values * 10, label = "RPM")
        ax[3].scatter(warning,df["RPM_SURF"].loc[warning]*10, color = "C3")
        ax[3].set_ylabel("[Rotations / 60s]")
        ax[3].legend()
        ax[4].plot(x_vals_for_plot,df["ropav"].values, label = "ROP")
        ax[4].scatter(warning,df["ropav"].loc[warning], color = "C3")
        ax[4].set_ylabel("[m / s]")
        ax[4].legend()
        ax[5].plot(x_vals_for_plot, df["HKLD"], label = "Hook load")
        ax[5].set_ylabel("[N]")
        ax[4].legend()
        ax[5].legend()
        ax[6].plot(x_vals_for_plot, df["DEPTH_BIT"], label = "Bit depth")
        ax[6].set_ylabel("[m]")
        ax[6].legend()
        ax[7].plot(x_vals_for_plot, df["DEPTH_HOLE"], label = "Hole depth")
        ax[7].set_ylabel("[m]")
        ax[7].legend()
        fig.text(0.5, 0.04, "Time [10s]", ha = "center", va = "center")


# Function maps creates a map between depth values and time values for convenient data-extraction
def map_pars_t_z(df, casing_shoe_md, md_resol, tmin, tmax, dt):
    """
    This function considers the depth flat output from the PBM model, as formatted above.
    It has 38 columns, with:

    - col0: time
    - col1: segment (1 = DS, 2 = OH, 3 = Ann)
    - col2 : mdin (m)
    - col3 = mdout (m)
    - col4-37: par to be mapped

    Returns:
         A 3d array (MD * time * 38 parameters). Time steps is equal to the one in the
         model. md step is an input parameter
    """
    # Returns all the time steps:
    time_steps = df["time"].unique()

    # Finds max MD
    md_max = df["mdIn"].max()

    # creates output array and defines dimensions:
    nt = time_steps.shape[0]
    md_out = np.arange(md_resol / 2.0, md_max, md_resol)

    nmd = md_out.shape[0]
    npar = df.shape[1]
    output_array = np.zeros((nt, nmd, npar))
    data_reinterp = np.zeros((nmd, npar))
    istep = 0

    # loop on time steps
    for tstep in time_steps:
        # extracts subtable: at tstep and in the annulus (loc > 1)
        data = df[df["time"] == tstep]
        data = data[data["loc"] > 1].values[::-1, :]
        # values are assigned to the center of the cell
        md_in = 0.5 * (data[:, 3] + data[:, 2])
        # output_grid
        # md_out = np.arange(md_resol/2.,md_max,md_resol)
        ind_md_out_max = np.int32(data[-1, 2] // md_resol)

        # BHA
        data_bha = data[data[:, 1] == 3, :][-1:, :]
        bhatop = data_bha[0, 3]
        bhabot = data_bha[0, 2]
        ind_md_ann_max = np.int32(bhatop // md_resol)
        ind_md_bha_max = np.int32(bhabot // md_resol)

        # Reinterpolation (1D in MD)
        f = sc.interpolate.interp1d(md_in, data, axis=0, fill_value="extrapolate")
        data_reinterp[:ind_md_ann_max, :] = f(md_out[:ind_md_ann_max])

        # BHA
        data_reinterp[ind_md_ann_max:ind_md_bha_max, :] = data_bha

        # VOlume ahead of the bit to BH (if any)
        if ind_md_bha_max < ind_md_out_max:
            data_reinterp[ind_md_bha_max:ind_md_out_max, :] = f(
                md_out[ind_md_bha_max:ind_md_out_max]
            )

        # masks non drilled cells
        data_reinterp[ind_md_out_max:, :] = np.nan

        # Redefines columns 2 and 3 giving respectively the depth of top and base of the current cell
        data_reinterp[:, 2] = md_out - md_resol / 2.0
        data_reinterp[:, 3] = md_out + md_resol / 2.0

        output_array[istep, :, :] = data_reinterp
        istep += 1

        # we reshape the output to have para first, then time, then md

    # Now that all the depth grids are consistent, we can reinterpolate the time
    time_steps_final = np.arange(tmin, tmax, dt)
    X = time_steps
    A = output_array
    f = sc.interpolate.interp1d(X, A, axis=0, fill_value="extrapolate")
    A_reinterp = f(time_steps_final)

    return np.transpose(A_reinterp, [2, 0, 1])



def extract_time_series_from_depth_data_for_given_variables(depth_data, position_for_time_series,columns):
    """
    Function takes optional variables and a time-series for the position of a sensor, and makes
    a time series for the variables at the position in question.

    This allows us for instance to make a time series for the Reynolds number at
    the position of the ASM sensor.

    """
    
    depth_indices = np.array([depth // 10 if depth > 0 else 0 for depth in position_for_time_series], dtype = np.int16)
    num_time_stamps = len(depth_indices)

    
    values_for_position = {}
    for index in range(len(depth_data)):
        
        values_for_position[columns[index]] = np.array([depth_data[index,time_index,depth_indices[time_index]]  for time_index in range(num_time_stamps)])

    return values_for_position



def test_results_confmat(test_confmat, model_name):
    """
    Simple function to evaluate which model performs the best
    on the test set using the confusion matrix from the k fold
    algorithm.

    """
    test_accuracy = round((test_confmat[0,0] + test_confmat[1,1]) / np.sum(test_confmat),2)
    sensitivity = test_confmat[1,1] / (test_confmat[0,1] + test_confmat[1,1])
    specificity = test_confmat[0,0] / (test_confmat[0,0] + test_confmat[1,0])
    print(f'Testing for {model_name}..')
    print(test_confmat)
    print("Testing accuracy is: ",test_accuracy)
    print("Sensitivity is:", round(sensitivity,2), " and specificity is: ", round(specificity,2))
    print("")
    print("")


def define_threshold(slip_ratio,rel_bed_height, height_threshold):
    """
    Calculates a threshold value for the slip ratio based on the particle 
    bed height.
    """
    threshold_values = []
    for slip, height in zip(slip_ratio,rel_bed_height):
        if height >= height_threshold:
            threshold_values.append(slip)


    return max(threshold_values)


def get_herschel_bulkley_parameters_from_rheology_file(filepath):

    """
    Takes in the filepath to a Rheology.out file and returns a dataframe
    with the calculated Herschel Bulkley parameters for a given pressure and
    temperature.

    """
    df = pd.DataFrame(columns=["Temperature","Pressure","Consistency index (K)", "Yield stress (tau_y)", "Flow behaviour index (n)"])
    variables_for_df = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for line in lines:
        
        if 'iT, iP' in line.strip():
            temp_variables = []
            splited_line = line.split()
            temperature = np.float32(splited_line[-2])
            pressure = np.float32(splited_line[-1])
            temp_variables.append(temperature)
            temp_variables.append(pressure)
            
        if 'yp, k, m nl fit:' in line.strip():
            splited_line = line.split()

            yield_point = np.float32(splited_line[-3])
            consistency_index = np.float32(splited_line[-2])
            flow_behaviour_index = np.float32(splited_line[-1])
            
            temp_variables.append(yield_point)
            temp_variables.append(consistency_index)
            temp_variables.append(flow_behaviour_index)

            variables_for_df.append(temp_variables)

    for idx,variables in enumerate(variables_for_df):
        df.loc[idx] = variables
    return df


def get_testing_dataset_from_field_data(df,variables, indices,ecccentricty,inclination, fluid_type):
    """
    Taskes in a list of indices and creates datapoints for a number of variables for 
    the sensor data.

    This function kinda sucks. Making a copy.
    """
    df_of_variables = df[variables]
    df_of_variables = df_of_variables.reset_index(drop = True)
    df_of_variables_at_points = df_of_variables.loc[indices]
    if inclination != None:

        df_of_variables_at_points['inclination'] = inclination
    if ecccentricty != None:

        df_of_variables_at_points['eccentricity'] = ecccentricty
    if fluid_type != None:
        df_of_variables_at_points['fluid type'] = fluid_type

    return df_of_variables_at_points

def get_training_dataset_from_field_data_transients(df,df_sensor,df_simulated,variables,target, indices,window):
    """
    Taskes in a list of indices and creates datapoints for a number of variables for 
    the sensor data. Also returns measured and simulated ecd values for the other positons.
    This is hardcoded due to it always being neeeded.

    """
    df_of_variables = df[variables]
    df_of_variables = df_of_variables.reset_index(drop = True)
    features_as_numpy = np.zeros(shape=(len(indices), 2 * window, len(variables)))
    targets_as_numpy = np.zeros(shape=(len(indices), 2 * window))
    variables_from_original_plot = np.zeros(shape = (len(indices), 2 * window, 3))

    variables_from_original_plot[:,:, 0] = [df_sensor["ASMECD2-T"].iloc[idx - window : idx + window].to_numpy() for idx in indices]
    variables_from_original_plot[:,:, 1] = [df_simulated["ecdAtPos2"].iloc[idx - window : idx + window].to_numpy() for idx in indices]
    variables_from_original_plot[:,:, 2] = [df_simulated["ecdAtPos3"].iloc[idx - window : idx + window].to_numpy() for idx in indices]

    for index, variable in enumerate(variables):
        features_as_numpy[:,:, index] = np.array([df_of_variables[variable].iloc[idx - window: idx + window].to_numpy() for idx in indices])
    
    targets_as_numpy[:,:] = np.array([df_sensor[target].iloc[idx - window : idx + window].to_numpy() for idx in indices])

    return features_as_numpy, targets_as_numpy, variables_from_original_plot


def regression_for_asm_data_points(sensor_1, sensor_2, idx, number_of_points_to_evaluate, degree):
    """
    Function thats performs a regression on a series of datapoints from the asm_sensors.
    The aim is to see of a trend detection can be performed with the aim of detecting a pack
    of scenario.
    """
    assert idx >= number_of_points_to_evaluate,"Index needs to be larger than number_of_past_points"
    
    reg_points_sensor_1 = sensor_1[idx - number_of_points_to_evaluate:idx]
    reg_points_sensor_2 = sensor_2[idx - number_of_points_to_evaluate:idx]

    x_axis_for_reg = np.arange(0, number_of_points_to_evaluate, 1)
    
    reg_sensor_1 = np.polyfit(x_axis_for_reg,reg_points_sensor_1, deg = degree)
    reg_sensor_2 = np.polyfit(x_axis_for_reg,reg_points_sensor_2, deg = degree)

    best_fit_sensor_1 = np.poly1d(reg_sensor_1)
    best_fit_sensor_2 = np.poly1d(reg_sensor_2)

    return reg_sensor_1, reg_sensor_2, best_fit_sensor_1,best_fit_sensor_2,x_axis_for_reg, reg_points_sensor_1, reg_points_sensor_2

def plot_flow_rpm_mud_dens_out(df_sensor,idx, *transport_times, flow_and_rpm = False):
    """
    Function that plots flow, rpm and mud density out. The mud density out is plotted
    with a shaded region where the expected arrival time of the cutting is marked.
    """
    timehorizon = transport_times[-1]
    flow_to_consider_for_transport =df_sensor["flowinav"].iloc[idx:idx + timehorizon].values 
    rpm_to_consider_for_transport = 10*df_sensor["rpmav"].iloc[idx:idx + timehorizon].values
    mud_density_out_to_consider_for_transport = df_sensor["MUD_DENS_OUT"].iloc[idx:idx + timehorizon].values
    transport_region = np.arange(transport_times[0], transport_times[-1],1)
    before_transport_region = np.arange(0,transport_times[0],1)

    average_time_derivative_transport_region = np.round(np.mean(np.diff(mud_density_out_to_consider_for_transport[transport_region-1])),4)
    average_time_derivative_before_transport_region = np.round(np.mean(np.diff(mud_density_out_to_consider_for_transport[before_transport_region-1])),4)

    if flow_and_rpm:
        plt.figure(figsize=(20,10))
        plt.title(f"Flow rate in after transient at index {idx}")
        plt.plot(flow_to_consider_for_transport)
        plt.ylabel("$q$ [m^3 / s]")
        plt.xlabel("Time [10s]")
        plt.show()

        plt.figure(figsize=(20,10))
        plt.title(f"Rotation rate after transient at index {idx}")
        plt.plot(rpm_to_consider_for_transport)

        plt.ylabel("RPM")
        plt.xlabel("Time [10s]")
        plt.show()

    plt.figure(figsize=(20,10))
    plt.title(f"Mud density out after transient at index {idx}", fontsize = 24)
    plt.ylabel(r"$\rho$ [kg / m^3]",fontsize = 22)
    plt.xlabel("Time [10s]",fontsize = 22)
    plt.plot(mud_density_out_to_consider_for_transport)
    plt.fill_between(transport_region,np.min(mud_density_out_to_consider_for_transport[transport_region]), 
                     np.max(mud_density_out_to_consider_for_transport[transport_region]), color = "red", alpha = 0.3, 
                     label =f" {r'$\overline{\rho}$'} = {np.round(np.mean(mud_density_out_to_consider_for_transport[transport_region]),2)} kg / m^3 and mean d{r'$\rho$'} / dt = {average_time_derivative_transport_region} kg /s m^3")
    plt.fill_between(before_transport_region,np.min(mud_density_out_to_consider_for_transport[before_transport_region]),
                      np.max(mud_density_out_to_consider_for_transport[before_transport_region]), color = "lightblue", alpha = 0.3, 
                      label =f" {r'$\overline{\rho}$'} = {np.round(np.mean(mud_density_out_to_consider_for_transport[before_transport_region]),2)} kg / m^3 and mean d{r'$\rho$'} / dt = {average_time_derivative_before_transport_region} kg /s m^3")
    plt.legend(fontsize = 14)
    plt.show()
   



def calculate_cross_correlation(list_1, list_2):
    """
    Takes two lists and computes the  correlation of list 1 shifted n_time_steps over list 2.
    Think of a sliding window of length list_1 goes over list_2 and computes the correlation of
    the window size. This will be appended to a list of size len(list_2) - window_size. giving the laged 
    correlation / cross correlation

    Returnr the cross_correlatin matrix for every lag, the max_covariance_value, corresponding index, and the p_value accociated with this corelation
    """
    assert len(list_2)>=len(list_1)
    window_size = len(list_1)
    cross_correlation = []
    max_corr = 0
    for idx in range(len(list_2)- window_size):

        correlation = np.cov(list_1, list_2[idx:idx + window_size])
        temp_corr = correlation[0,1]
        if temp_corr > max_corr:
            max_corr = temp_corr
            max_idx = idx
            corr,p_value = sc.stats.pearsonr(list_1, list_2[idx: idx + window_size])
        cross_correlation.append(correlation)
    return np.array(cross_correlation), max_corr, max_idx, p_value

