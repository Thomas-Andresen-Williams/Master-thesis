import numpy as np

class PackOffDetectionAlgorithm:
    """
    This class inocorperates the funcitonality of the patent for detecting poor hole cleaning
    from https://patents.google.com/patent/US6401838B1/en
    Uses different functions to raise different types of warnings for detecting poor
    hole cleaning.

    calculate_differences_and_raise_warning uses the difference between ecd sensors to detect
    'pack off tendencies between ecd 1 and ecd 2 sensor
    """
    def __init__(self, df_sensor, df_sim_obs,df_sim = None, variables = None):
        
        self.time_stamps = df_sensor.index
        self.ecd_downhole = df_sensor["DH_PRESS_ECD"].values
        self.asm_ecd_1 = df_sensor["ASMECD1-T"].values
        self.asm_ecd_2 = df_sensor["ASMECD2-T"].values
        self.surface_torque = df_sensor["TORQ"].values
        self.rop = df_sensor["ROP"].values
        self.sim_ecd_1 = df_sim_obs["ecdAtPos1"].values
        self.sim_ecd_2 = df_sim_obs["ecdAtPos2"].values

        self.warnings_cumsum = []
        self.warnings_average_slope = []
      
    
  

    

    def test_cumsum_on_sensor_diff(self,idx,number_of_points_forward,cumsum_threshold):
        """
        This dunction takes in a specified index, and a time horizon alongside a threshold
        that is there to determine when an alarm is supposed to be raised.

        It calculates the difference between the sensor positions for the sensor values
        and for the simulated values. If the difference in cumulative sum between sensors and 
        simulated data is bigger than a certain theshold it raises an alarm.

        """
        diffs_asm = self.asm_ecd_1[idx:idx + number_of_points_forward] - self.asm_ecd_2[idx:idx + number_of_points_forward]
        diffs_sim = self.sim_ecd_1[idx:idx + number_of_points_forward] - self.sim_ecd_2[idx:idx + number_of_points_forward]
        cum_asm = np.cumsum(diffs_asm)
        cum_sim = np.cumsum(diffs_sim)

        if (cum_asm[-1] - cum_sim[-1]) > cumsum_threshold:
            self.warnings_cumsum.append([self.time_stamps[idx],idx])
    
    def test_positive_and_negative_slope(self,idx,number_of_time_steps_forward, threshold):
        """
        This function calculates the average slope for the ecd measurements for a desired horizon and raises and alarm if the there
        is a difference in average slope between the sensors. This method leverages the simulated data to determine if the values are above a certain
        theshold.
        """
        asm1 = self.asm_ecd_1[idx:idx + number_of_time_steps_forward]
        asm2 = self.asm_ecd_2[idx:idx + number_of_time_steps_forward]
        sim1 = self.sim_ecd_1[idx:idx + number_of_time_steps_forward]
        sim2 = self.sim_ecd_2[idx:idx + number_of_time_steps_forward]
        

        average_slope_asm1 = (asm1[-1]-asm1[0]) / len(asm1)
        average_slope_asm2 = (asm2[-1]-asm2[0]) / len(asm2)
        average_slope_sim1 = (sim1[-1]-sim1[0]) / len(sim1)
        average_slope_sim2 = (sim2[-1]-sim2[0]) / len(sim2)
        # Imposing a restriction that the ROP needs to be larger than zero
        if average_slope_asm1 > 0 and average_slope_asm2 < threshold / 10 and average_slope_asm1 - average_slope_sim1 > threshold and self.rop[idx] > 0:
            self.warnings_average_slope.append([self.time_stamps[idx],idx])
