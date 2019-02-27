# Note: Should the agent learn to go to destination by learning?
def goto_dest(self):
    # set threshold for confirming the correct heading towards destination
    threshold = deg2rad(8.)
    heading_vec = np.array([np.cos(self.heading), np.sin(self.heading), 0])
    desti_dir_angle = norm_angle(atan2(self.y_dest - self.y, self.x_dest - self.x))
    heading_error = np.abs(desti_dir_angle - self.heading)
    if heading_error < threshold:
        self.heading = desti_dir_angle
    else: 
        dest_vec = np.array([np.cos(desti_dir_angle), np.sin(desti_dir_angle), 0])
        cross_result = np.cross(heading_vec, dest_vec)

    if cross_result[-1] < 0:
        side = 'Right'
    elif cross_result[-1] > 0:
        side = 'Left'
    else:
        side = 'OnDir'
    # get turn rate
    if side == 'Right':
        self.turn_rate = - MAX_TURN_RATE
    elif side == 'Left':
        self.turn_rate = MAX_TURN_RATE
    else:
        self.turn_rate = 0 # keep heading
    self.v = MAX_V
    self.heading = norm_angle(self.heading + self.turn_rate * DT)
    # update coordinates
    self.x += np.cos(self.heading) * self.v * DT
    self.y += np.sin(self.heading) * self.v * DT
    self.prev_dist_to_dest = self.dist_to_dest
    self.dist_to_dest = np.sqrt((self.dest_y - self.y)**2 + (self.dest_x - self.x)**2)