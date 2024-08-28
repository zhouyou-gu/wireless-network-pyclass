import math
import numpy as np

class Trajectory:
    def __init__(self, id, name, altitude, inclination, raan, earth_radius=6371):
        self.id = id
        self.name = name
        self.altitude = altitude  # km above Earth's surface
        self.inclination = math.radians(inclination)  # Convert to radians
        self.raan = math.radians(raan)  # Right Ascension of the Ascending Node in radians
        self.earth_radius = earth_radius  # Earth's radius in km
        self.semi_major_axis = self.altitude + self.earth_radius  # Semi-major axis in km
        self.mu = 398600  # Earth's gravitational parameter, km^3/s^2
        self.angular_velocity_earth = 7.2921159e-5  # Earth's angular velocity in rad/s
        
        self.time_elapsed_second = 0

    def orbital_period(self):
        """ Calculate the orbital period of the satellite in minutes """
        T = 2 * math.pi * math.sqrt(self.semi_major_axis**3 / self.mu)  # Period in seconds
        return T / 60  # Convert to minutes

    def position_in_orbit(self):
        """ Calculate the satellite's position in its orbital plane """
        n = math.sqrt(self.mu / self.semi_major_axis**3)  # Mean motion (rad/s)
        M = n * self.time_elapsed_second  # Mean anomaly
        E = M  # For circular orbits (eccentricity e ~ 0), E ≈ M
        true_anomaly = 2 * math.atan2(math.sqrt(1) * math.sin(E/2), math.cos(E/2))  # True anomaly

        # Orbital radius (for circular orbit, it's constant)
        r = self.semi_major_axis

        # Position in orbital plane (ignoring eccentricity for simplicity)
        x_orbit = r * math.cos(true_anomaly)
        y_orbit = r * math.sin(true_anomaly)
        return x_orbit, y_orbit

    def ground_track(self):
        """ Calculate the satellite's ground track (latitude and longitude) on Earth """
        x_orbit, y_orbit = self.position_in_orbit()

        # Transform orbital plane coordinates to Earth coordinates considering inclination and RAAN
        x_earth = x_orbit * math.cos(self.raan) - y_orbit * math.cos(self.inclination) * math.sin(self.raan)
        y_earth = x_orbit * math.sin(self.raan) + y_orbit * math.cos(self.inclination) * math.cos(self.raan)
        z_earth = y_orbit * math.sin(self.inclination)

        # Longitude of the satellite (considering Earth's rotation)
        lon = math.atan2(y_earth, x_earth) - self.angular_velocity_earth * self.time_elapsed_second
        lon = math.degrees(lon) % 360

        # Latitude of the satellite
        lat = math.degrees(math.asin(z_earth / self.semi_major_axis))
        
        # Normalize longitude to [-180, 180]
        if lon > 180:
            lon -= 360

        return lat, lon

    def step(self, time_step_second):
        self.time_elapsed_second += time_step_second

    def simulate_n_points(self, time_step_second, n_step):
        positions = []
        for t in range(n_step):
            self.step(time_step_second)
            lat, lon = self.ground_track()
            positions.append((lat, lon))
        return np.asarray(positions)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Assuming `satellite` is an instance of a class with a `simulate_n_points` method
    satellite = Trajectory(0, "Starlink-1", altitude=550, inclination=50, raan=0)
    positions = satellite.simulate_n_points(60, 100)
    lats, lons = zip(*positions)
    
    # Convert longitudes to handle crossing of the dateline (-180 to 180 degrees)
    lons = np.array(lons)
    lats = np.array(lats)
    lons[lons > 180] -= 360  # Wrap around longitudes greater than 180

    # Set up the map with Cartopy
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_global()

    # Add features to the map
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='aqua')
    ax.add_feature(cfeature.LAKES, facecolor='blue')
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')


    # Plot the satellite trajectory, ensuring continuous lines
    for i in range(len(lons) - 1):
        if abs(lons[i+1] - lons[i]) > 180:
            # Handle the discontinuity by breaking the line
            print(lons[i+1] , lons[i])
            ax.plot([lons[i], np.sign(lons[i]) * 180], [lats[i], lats[i]], color='red', linewidth=2, marker='o', markersize=4)
            ax.plot([np.sign(lons[i+1]) * 180, lons[i+1]], [lats[i+1], lats[i+1]], color='red', linewidth=2, marker='o', markersize=4)
        else:
            ax.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]], color='red', linewidth=2, marker='o', markersize=4)

    print(lons[:-1:5],lats[:-1:5])
    arrow_skip = 5  # Define how frequently to place arrows (every 5th point)
    U = np.diff(lons[::arrow_skip])
    V = np.diff(lats[::arrow_skip])

    # Add arrows to indicate the direction of the satellite
    # Use quiver to add arrows
    arrow_skip = 5  # Define how frequently to place arrows (every 5th point)
    ax.quiver(lons[::arrow_skip][:-1], lats[::arrow_skip][:-1],   # Positions
            U, V,  # Directions
            scale_units='xy', angles='xy', scale=1, color='blue', transform=ccrs.PlateCarree())


    # Add title and show plot
    ax.set_title('Satellite Trajectory on 2D Earth Texture')
    plt.show()