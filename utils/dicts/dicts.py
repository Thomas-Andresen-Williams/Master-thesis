dict_of_variables = {
    "Slip ratio" : df["Particle slip ratio in test section"].astype("float32"),
    # "Eccentricity" : df["Eccentricity"].astype("float32"),
    # "Inclination" : df["Inclination"],
    "Reynolds" : df["Reynolds annulus (no cuttings)"].astype("float32"),
    "Taylor" : df["Taylor number"].astype("float32"),
    "Shields" : df["Shields number"].astype("float32"),
    # "Froude" : df["Froude number"].astype("float32"),
    # "Bingham" : df["Bingham number"].astype("float32"),
    # "Fluid name": df["Fluid name"],
    # 'Friction fac': df["Friction factor mod annulus"].astype("float32")
    # "Friction coef" : df["mean fricCoeff"].astype("float32"),
}