"""
Modified version of code from https://github.com/ika-rwth-aachen/drone-dataset-tools
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from matplotlib.widgets import Button, Slider
from loguru import logger

from igp2.recognition.goalrecognition import GoalRecognition
from igp2.recognition.goalprobabilities import GoalsProbabilities
from igp2.results import *


class TrackVisualizer(object):
    def __init__(self, config, tracks, static_info, meta_info, result : EpisodeResult, fig=None,
                 scenario=None, episode=None):
        self.config = config
        self.input_path = config["input_path"]
        self.recording_name = config["recording_name"]
        self.image_width = None
        self.image_height = None
        self.scale_down_factor = config["scale_down_factor"]
        self.skip_n_frames = config["skip_n_frames"]
        self.scenario = scenario
        self.episode = episode
        self.result = dict(result.data)

        # set up goal recognition #! REMOVE
        self.all_agents_probabilities = {}

        # Get configurations
        if self.scale_down_factor % 2 != 0:
            logger.warning("Please only use even scale down factors!")

        # Tracks information
        self.tracks = tracks
        self.static_info = static_info
        self.meta_info = meta_info
        self.maximum_frames = np.max([self.static_info[track["trackId"]]["finalFrame"] for track in self.tracks])

        # Save ids for each frame
        self.ids_for_frame = {}
        for i_frame in range(self.maximum_frames):
            indices = [i_track for i_track, track in enumerate(self.tracks)
                       if
                       self.static_info[track["trackId"]]["initialFrame"] <= i_frame <= self.static_info[track["trackId"]][
                           "finalFrame"]]
            self.ids_for_frame[i_frame] = indices

        # Initialize variables
        self.current_frame = 0
        self.changed_button = False
        self.rect_map = {}
        self.plotted_objects = []
        self.track_info_figures = {}
        self.y_sign = 1

        # Create figure and axes
        if fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self.fig.set_size_inches(18, 8)
            plt.subplots_adjust(left=0.0, right=1.0, bottom=0.20, top=0.99)
        else:
            self.fig = fig
            self.ax = self.fig.gca()

        self.fig.canvas.set_window_title("Recording {}".format(self.recording_name))

        # Check whether to use the given background image
        background_image_path = self.config["background_image_path"]
        if background_image_path is not None:
            self.background_image = skimage.io.imread(background_image_path)
            self.image_height = self.background_image.shape[0]
            self.image_width = self.background_image.shape[1]
            self.ax.imshow(self.background_image)
        else:
            self.background_image = np.zeros((1700, 1700, 3), dtype=np.float64)
            self.image_height = 1700
            self.image_width = 1700
            self.ax.imshow(self.background_image)

        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # Dictionaries for the style of the different objects that are visualized
        self.rect_style = dict(fill=True, edgecolor="k", alpha=0.4, zorder=19)
        self.triangle_style = dict(facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6, zorder=19)
        self.text_style = dict(picker=True, size=8, color='k', zorder=20, ha="center")
        self.text_box_style = dict(boxstyle="round,pad=0.2", alpha=.6, ec="black", lw=0.2)
        self.track_style = dict(linewidth=1, zorder=10)
        self.centroid_style = dict(fill=True, edgecolor="black", lw=0.1, alpha=1,
                                   radius=5, zorder=30)
        self.track_style_future = dict(color="linen", linewidth=1, alpha=0.7, zorder=10)
        self.track_style_optstart = dict(color="cyan", linewidth=1.5, alpha=0.7, zorder=10)
        self.track_style_optcurr = dict(color="g", linewidth=1, alpha=0.7, zorder=10)
        self.circ_stylel = dict(facecolor="g", fill=True, edgecolor="r", lw=0.1, alpha=1,
                                radius=8, zorder=30)
        self.circ_styler = dict(facecolor="b", fill=True, edgecolor="r", lw=0.1, alpha=1,
                                radius=8, zorder=30)
        self.colors = dict(car="blue", truck_bus="orange", pedestrian="red", bicycle="yellow", default="green")

        # Initialize the plot with the bounding boxes of the first frame
        self.update_figure()

        ax_color = 'lightgoldenrodyellow'
        # Define axes for the widgets
        self.ax_slider = self.fig.add_axes([0.2, 0.035, 0.2, 0.04], facecolor=ax_color)  # Slider
        self.ax_button_previous2 = self.fig.add_axes([0.02, 0.035, 0.06, 0.04])
        self.ax_button_previous = self.fig.add_axes([0.09, 0.035, 0.06, 0.04])
        self.ax_button_next = self.fig.add_axes([0.44, 0.035, 0.06, 0.04])
        self.ax_button_next2 = self.fig.add_axes([0.51, 0.035, 0.06, 0.04])
        self.ax_button_play = self.fig.add_axes([0.58, 0.035, 0.06, 0.04])
        self.ax_button_stop = self.fig.add_axes([0.65, 0.035, 0.06, 0.04])

        # Define the widgets
        self.frame_slider = DiscreteSlider(self.ax_slider, 'Frame', 0, self.maximum_frames-1, valinit=self.current_frame,
                                           valfmt='%s')
        self.button_previous2 = Button(self.ax_button_previous2, 'Previous x%d' % self.skip_n_frames)
        self.button_previous = Button(self.ax_button_previous, 'Previous')
        self.button_next = Button(self.ax_button_next, 'Next')
        self.button_next2 = Button(self.ax_button_next2, 'Next x%d' % self.skip_n_frames)
        self.button_play = Button(self.ax_button_play, 'Play')
        self.button_stop = Button(self.ax_button_stop, 'Stop')

        # Define the callbacks for the widgets' actions
        self.frame_slider.on_changed(self.update_slider)
        self.button_previous.on_clicked(self.update_button_previous)
        self.button_next.on_clicked(self.update_button_next)
        self.button_previous2.on_clicked(self.update_button_previous2)
        self.button_next2.on_clicked(self.update_button_next2)
        self.button_play.on_clicked(self.start_play)
        self.button_stop.on_clicked(self.stop_play)

        self.timer = self.fig.canvas.new_timer(interval=25*self.skip_n_frames)
        self.timer.add_callback(self.update_button_next2, self.ax)

        # Define the callbacks for the widgets' actions
        self.fig.canvas.mpl_connect('key_press_event', self.update_keypress)

        self.ax.set_autoscale_on(False)

        self.scenario.plot_goals(self.ax, self.meta_info["orthoPxToMeter"] * self.scale_down_factor, flipy=True)

    def update_keypress(self, evt):
        if evt.key == "right" and self.current_frame + self.skip_n_frames < self.maximum_frames:
            self.current_frame = self.current_frame + self.skip_n_frames
            self.trigger_update()
        elif evt.key == "left" and self.current_frame - self.skip_n_frames >= 0:
            self.current_frame = self.current_frame - self.skip_n_frames
            self.trigger_update()

    def update_slider(self, value):
        if not self.changed_button:
            self.current_frame = value
            self.trigger_update()
        self.changed_button = False

    def update_button_next(self, _):
        if self.current_frame + 1 < self.maximum_frames:
            self.current_frame = self.current_frame + 1
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning(
                "There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_next2(self, _):
        if self.current_frame + self.skip_n_frames < self.maximum_frames:
            self.current_frame = self.current_frame + self.skip_n_frames
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning("There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_previous(self, _):
        if self.current_frame - 1 >= 0:
            self.current_frame = self.current_frame - 1
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning("There are no frames available with an index lower than 1.")

    def update_button_previous2(self, _):
        if self.current_frame - self.skip_n_frames >= 0:
            self.current_frame = self.current_frame - self.skip_n_frames
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning("There are no frames available with an index lower than 1.")

    def start_play(self, _):
        self.timer.start()

    def stop_play(self, _):
        self.timer.stop()

    def trigger_update(self):
        self.remove_patches()
        self.update_figure()
        self.update_pop_up_windows()
        self.frame_slider.update_val_external(self.current_frame)
        self.fig.canvas.draw_idle()

    def update_figure(self):
        # Plot the bounding boxes, their text annotations and direction arrow
        plotted_objects = []
        for track_ind in self.ids_for_frame[self.current_frame]:
            track = self.tracks[track_ind]

            track_id = track["trackId"]
            static_track_information = self.static_info[track_id]
            initial_frame = static_track_information["initialFrame"]
            current_index = self.current_frame - initial_frame

            object_class = static_track_information["class"]
            is_vehicle = object_class in ["car", "truck_bus", "motorcycle"]
            bounding_box = track["bboxVis"][current_index] / self.scale_down_factor
            center_points = track["centerVis"] / self.scale_down_factor
            center_point = center_points[current_index]

            color = self.colors[object_class] if object_class in self.colors else self.colors["default"]

            # Obtain result data
            frame_result, _, _ = self.get_frame_result(track_id)

            if self.config["plotBoundingBoxes"] and is_vehicle:
                rect = plt.Polygon(bounding_box, True, facecolor=color, **self.rect_style)
                self.ax.add_patch(rect)
                plotted_objects.append(rect)

            if self.config["plotDirectionTriangle"] and is_vehicle:
                # Add triangles that display the direction of the cars
                triangle_factor = 0.75
                a_x = bounding_box[3, 0] + ((bounding_box[2, 0] - bounding_box[3, 0]) * triangle_factor)
                b_x = bounding_box[0, 0] + ((bounding_box[1, 0] - bounding_box[0, 0]) * triangle_factor)
                c_x = bounding_box[2, 0] + ((bounding_box[1, 0] - bounding_box[2, 0]) * 0.5)
                triangle_x_position = np.array([a_x, b_x, c_x])

                a_y = bounding_box[3, 1] + ((bounding_box[2, 1] - bounding_box[3, 1]) * triangle_factor)
                b_y = bounding_box[0, 1] + ((bounding_box[1, 1] - bounding_box[0, 1]) * triangle_factor)
                c_y = bounding_box[2, 1] + ((bounding_box[1, 1] - bounding_box[2, 1]) * 0.5)
                triangle_y_position = np.array([a_y, b_y, c_y])

                # Differentiate between vehicles that drive on the upper or lower lanes
                triangle_info = np.array([triangle_x_position, triangle_y_position])
                polygon = plt.Polygon(np.transpose(triangle_info), True, **self.triangle_style)
                self.ax.add_patch(polygon)
                plotted_objects.append(polygon)

            if self.config["plotTrackingLines"]:
                plotted_centroid = plt.Circle((center_points[current_index][0],
                                               center_points[current_index][1]),
                                              facecolor=color, **self.centroid_style)
                self.ax.add_patch(plotted_centroid)
                plotted_objects.append(plotted_centroid)
                if center_points.shape[0] > 0:
                    # Calculate the centroid of the vehicles by using the bounding box information
                    # Check track direction
                    plotted_centroids = self.ax.plot(
                        center_points[0:current_index+1][:, 0],
                        center_points[0:current_index+1][:, 1],
                        color=color, **self.track_style)
                    plotted_objects.append(plotted_centroids)
                    if self.config["plotFutureTrackingLines"]:
                        # Check track direction
                        plotted_centroids_future = self.ax.plot(
                            center_points[current_index:][:, 0],
                            center_points[current_index:][:, 1],
                            **self.track_style_future)
                        plotted_objects.append(plotted_centroids_future)

            if self.config["showTextAnnotation"]:
                # Plot the text annotation
                annotation_text = "ID{}".format(track_id)
                if self.config["showClassLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    annotation_text += "{}".format(object_class[0])
                if self.config["showVelocityLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    current_velocity = np.sqrt(
                        track["xVelocity"][current_index] ** 2 + track["yVelocity"][current_index] ** 2) * 3.6
                    current_velocity = abs(float(current_velocity))
                    annotation_text += "{:.2f}km/h".format(current_velocity)
                if self.config["showRotationsLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    current_rotation = track["heading"][current_index]
                    annotation_text += "Deg%.2f" % current_rotation
                if self.config["showAgeLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    age = static_track_information["age"]
                    annotation_text += "Age%d/%d" % (current_index + 1, age)
                if frame_result is not None:
                    for goal_idx, ((goal, goal_type), prob) in enumerate(frame_result.goals_probabilities.items()):
                        if prob > 0:
                            annotation_text += '\nG{}: {:.3f}'.format(goal_idx, prob)
                        
                # Differentiate between using an empty background image and using the virtual background
                target_location = (
                    center_point[0],
                    (center_point[1]))
                text_location = (
                    (center_point[0] + 45),
                    (center_point[1] - 20))
                text_patch = self.ax.annotate(annotation_text, xy=target_location, xytext=text_location,
                                              bbox={"fc": color, **self.text_box_style}, **self.text_style)
                plotted_objects.append(text_patch)

        # Add listener to figure
        self.fig.canvas.mpl_connect('pick_event', self.on_click)
        # Save the plotted objects in a list
        self.plotted_objects = plotted_objects

    def get_frame_result(self, track_id):

        frame_result = None
        agent_result = None
        closest_frame = None
        if track_id in self.episode.agents:
            try:
                agent_result = self.result[track_id]
            except KeyError:
                logger.info("Could not find agent {} in result binary, will not display result data.", track_id)
            if agent_result is not None:
                agent_datadict = dict([datum[0:2] for datum in agent_result.data])
                try:
                    closest_frame = self.current_frame
                    frame_result = agent_datadict[closest_frame]
                except KeyError:
                    closest_frame = min(agent_datadict.keys(), key=lambda k: abs(k-self.current_frame))
                    frame_result = agent_datadict[closest_frame]
                    logger.debug("Frame {} for agent {} was not computed in result binary, will display result data for closest frame {}.", self.current_frame, track_id, closest_frame)

        return frame_result, closest_frame, agent_result

    def on_click(self, event):
        artist = event.artist
        text_value = artist._text
        if "ID" not in text_value:
            return

        try:
            track_id_string = text_value[:text_value.index("|")]
            track_id = int(track_id_string[2:])
            track = None
            for track in self.tracks:
                if track["trackId"] == track_id:
                    track = track
                    break
            if track is None:
                logger.error("No track with the ID {} was found. Nothing to show.".format(track_id))
                return
            static_information = self.static_info[track_id]
            #get agent result for closest frame to current frame
            frame_result, frame_id, agent_result = self.get_frame_result(track_id)

            # Create a new figure that pops up
            fig = plt.figure(np.random.randint(0, 5000, 1))
            fig.canvas.mpl_connect('close_event', lambda evt: self.close_track_info_figure(evt, track_id))
            fig.canvas.mpl_connect('resize_event', lambda evt: fig.tight_layout())
            fig.set_size_inches(12, 7)
            fig.canvas.set_window_title("Recording {}, Track {} ({}), Frame {}".format(self.recording_name,
                                                                             track_id, static_information["class"], frame_id))

            borders_list = []
            subplot_list = []

            subplot_index = 421

            optimum_trajectory = list(frame_result.optimum_trajectory.values())[agent_result.true_goal]
            x2 = optimum_trajectory.pathlength

            # ---------- Time elapsed ----------
            y2 = optimum_trajectory.times
            title = "Time [s]"
            subplot, y_limits = self.init_subplot(subplot_index, title, x2, y2)
            subplot_list.append(subplot)
            borders_list.append(y_limits)
            subplot_index = subplot_index + 1

            # ---------- Velocity ----------
            y2 = optimum_trajectory.velocity
            title = "Velocity [m/s]"
            subplot, _ = self.init_subplot(subplot_index, title, x2, y2)
            y_limits = [0, self.episode.metadata.max_speed]
            subplot_list.append(subplot)
            borders_list.append(y_limits)
            subplot_index = subplot_index + 1

            # ---------- Heading ----------
            y2 = np.rad2deg(np.unwrap(optimum_trajectory.heading))
            title = "Heading [deg]"
            subplot, y_limits = self.init_subplot(subplot_index, title, x2, y2)
            #y_limits = [-190, 1]
            subplot_list.append(subplot)
            borders_list.append(y_limits)
            subplot_index = subplot_index + 1

            # ---------- Acceleration ----------
            y2 = optimum_trajectory.acceleration
            title = "Acceleration [m/s2]"
            subplot, y_limits = self.init_subplot(subplot_index, title, x2, y2)
            subplot_list.append(subplot)
            borders_list.append(y_limits)
            subplot_index = subplot_index + 1

            # ---------- Angular Velocity ----------
            y2 = np.rad2deg(optimum_trajectory.angular_velocity)
            title = "Angular Velocity [deg/s]"
            subplot, y_limits = self.init_subplot(subplot_index, title, x2, y2)
            subplot_list.append(subplot)
            borders_list.append(y_limits)
            subplot_index = subplot_index + 1

            # ---------- Jerk ----------
            y2 = optimum_trajectory.jerk
            title = "Jerk [m/s3]"
            subplot, y_limits = self.init_subplot(subplot_index, title, x2, y2)
            subplot_list.append(subplot)
            borders_list.append(y_limits)
            subplot_index = subplot_index + 1

            # ---------- Angular Acceleration ----------
            y2 = np.rad2deg(optimum_trajectory.angular_acceleration)
            title = "Angular Acceleration [deg/s2]"
            subplot, y_limits = self.init_subplot(subplot_index, title, x2, y2)
            subplot_list.append(subplot)
            borders_list.append(y_limits)
            subplot_index = subplot_index + 1

            # ---------- Curvature ----------
            y2 = optimum_trajectory.curvature
            title = "Curvature [1/m]"
            subplot, y_limits = self.init_subplot(subplot_index, title, x2, y2)
            subplot_list.append(subplot)
            borders_list.append(y_limits)
            subplot_index = subplot_index + 1

            self.track_info_figures[track_id] = {"main_figure": fig,
                                                 "borders": borders_list,
                                                 "subplots": subplot_list}

            self.update_pop_up_windows()
            plt.show()
        except Exception:
            logger.exception("An exception occured trying to display track information")
            return

    def close_track_info_figure(self, evt, track_id):
        if track_id in self.track_info_figures:
            self.track_info_figures[track_id]["main_figure"].canvas.mpl_disconnect('close_event')
            self.track_info_figures.pop(track_id)

    def get_figure(self):
        return self.fig

    def remove_patches(self):
        self.fig.canvas.mpl_disconnect('pick_event')
        for figure_object in self.plotted_objects:
            if isinstance(figure_object, list):
                figure_object[0].remove()
            else:
                figure_object.remove()
        self.plotted_objects = []

    def update_pop_up_windows(self):

        for track_id, track_map in self.track_info_figures.items():

            #get agent result for closest frame to current frame
            frame_result, frame_id, agent_result = self.get_frame_result(track_id)

            #obtain the index in the trajectory corresponding to current displayed results
            static_information = self.static_info[track_id]
            initial_frame = static_information["initialFrame"]
            current_index = frame_id - initial_frame

            #plot the IGP2 generated trajectories for the selected agent on the main plot
            if self.config["showOptStartTrajectory"]:
                if frame_result is not None:
                    for opt_trajectory in frame_result.optimum_trajectory.values():
                        if opt_trajectory is not None:
                            x_opt_px = opt_trajectory.path[:,0] / self.meta_info["orthoPxToMeter"] / self.scale_down_factor
                            y_opt_px = - opt_trajectory.path[:,1] / self.meta_info["orthoPxToMeter"] / self.scale_down_factor
                            plotted_OptStartTrajectory = self.ax.plot(x_opt_px,  y_opt_px, **self.track_style_optstart)
                            self.plotted_objects.append(plotted_OptStartTrajectory)

            if self.config["showOptCurrentTrajectory"]:
                if frame_result is not None:
                    for goal_idx, ((goal, goal_type), curr_trajectory) in enumerate(frame_result.current_trajectory.items()):
                        if curr_trajectory is not None and frame_result.goals_probabilities[(goal, goal_type)] != 0.0:
                            x_curr_px = curr_trajectory.path[:,0] / self.meta_info["orthoPxToMeter"] / self.scale_down_factor
                            y_curr_px = - curr_trajectory.path[:,1] / self.meta_info["orthoPxToMeter"] / self.scale_down_factor
                            plotted_OptCurrentTrajectory = self.ax.plot(x_curr_px,  y_curr_px, **self.track_style_optcurr)
                            self.plotted_objects.append(plotted_OptCurrentTrajectory)

            current_trajectory = list(frame_result.current_trajectory.values())[agent_result.true_goal]
            optimum_trajectory = list(frame_result.optimum_trajectory.values())[agent_result.true_goal]
            x1 = current_trajectory.pathlength
            x2 = optimum_trajectory.pathlength
            x_limits = [min(min(x1), min(x2)), max(max(x1), max(x2))]

            borders = track_map["borders"]
            subplots = track_map["subplots"]

            x1 = current_trajectory.pathlength
            subplot_index = 0

            ## Time
            y1 = current_trajectory.times
            self.update_subplot(subplots, borders, subplot_index, x1, y1)
            subplot_index +=1

            ## Velocity
            y1 = current_trajectory.velocity
            self.update_subplot(subplots, borders, subplot_index, x1, y1)
            subplot_index +=1

            ## Heading
            y1 = np.rad2deg(np.unwrap(current_trajectory.heading))
            self.update_subplot(subplots, borders, subplot_index, x1, y1)
            subplot_index +=1

            ## Acceleration
            y1 = current_trajectory.acceleration
            self.update_subplot(subplots, borders, subplot_index, x1, y1)
            subplot_index +=1

            ## Angular velocity
            y1 = np.rad2deg(current_trajectory.angular_velocity)
            self.update_subplot(subplots, borders, subplot_index, x1, y1)
            subplot_index +=1

            ## Jerk
            y1 = current_trajectory.jerk
            self.update_subplot(subplots, borders, subplot_index, x1, y1)
            subplot_index +=1

            ## Angular acceleration
            y1 = np.rad2deg(current_trajectory.angular_acceleration)
            self.update_subplot(subplots, borders, subplot_index, x1, y1)
            subplot_index +=1

            ## Curvature
            y1 = current_trajectory.curvature
            self.update_subplot(subplots, borders, subplot_index, x1, y1)
            subplot_index +=1

            for subplot_index, subplot_figure in enumerate(subplots):
                new_line = subplot_figure.plot([current_trajectory.pathlength[current_index], current_trajectory.pathlength[current_index]], borders[subplot_index], "--r")
                self.plotted_objects.append(new_line)
                subplot_figure.axes.set_xlim(x_limits)
            track_map["main_figure"].canvas.set_window_title("Recording {}, Track {} ({}), Frame {}".format(self.recording_name,
                                                                             track_id, static_information["class"], frame_id))
            track_map["main_figure"].canvas.draw_idle()

    def init_subplot(self, subplot_index, title, xdata, ydata):
        sub_plot = plt.subplot(subplot_index, title=title)
        plot = sub_plot.plot(xdata, ydata, **self.track_style_optstart) #! double check
        y_limits = [min(ydata), max(ydata)]
        offset = (y_limits[1] - y_limits[0]) * 0.05
        y_limits = [y_limits[0] - offset, y_limits[1] + offset]
        sub_plot.grid(True)
        plt.xlabel('Pathlength')

        return sub_plot, y_limits

    def update_subplot(self, subplots, borders, subplot_id, xdata, ydata):
        curr_plot = subplots[subplot_id].plot(xdata, ydata, **self.track_style_optcurr)
        self.plotted_objects.append(curr_plot)
        y_limits = [min(ydata), max(ydata)]
        offset = (y_limits[1] - y_limits[0]) * 0.05
        y_limits = [y_limits[0] - offset, y_limits[1] + offset]
        borders[subplot_id][0] = min(borders[subplot_id][0], y_limits[0])
        borders[subplot_id][1] = max(borders[subplot_id][1], y_limits[1])
        subplots[subplot_id].axes.set_ylim(y_limits)
        #subplots[subplot_id].axes.set_ylim(borders[subplot_id])

    @staticmethod
    @logger.catch(reraise=True)
    def show():
        plt.show()


class DiscreteSlider(Slider):
    def __init__(self, *args, **kwargs):
        self.inc = kwargs.pop('increment', 1)
        self.valfmt = '%s'
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        if self.val != val:
            discrete_val = int(int(val / self.inc) * self.inc)
            xy = self.poly.xy
            xy[2] = discrete_val, 1
            xy[3] = discrete_val, 0
            self.poly.xy = xy
            self.valtext.set_text(self.valfmt % discrete_val)
            if self.drawon:
                self.ax.figure.canvas.draw()
            self.val = val
            if not self.eventson:
                return
            for cid, func in self.observers.items():
                func(discrete_val)

    def update_val_external(self, val):
        self.set_val(val)