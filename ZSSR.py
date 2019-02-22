import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.gridspec import GridSpec
from configs import Config
from utils import *
from simplenet import simpleNet

class ZSSR:
    # Basic current state variables initialization / declaration
    kernel = None
    learning_rate = None
    hr_father = None
    lr_son = None
    sr = None
    sf = None
    gt_per_sf = None
    final_sr = None
    hr_fathers_sources = []

    # Output variables initialization / declaration
    reconstruct_output = None
    train_output = None
    output_shape = None

    # Counters and logs initialization
    iter = 0
    base_sf = 1.0
    base_ind = 0
    sf_ind = 0
    mse = []
    mse_rec = []
    interp_rec_mse = []
    interp_mse = []
    mse_steps = []
    loss = []
    learning_rate_change_iter_nums = []
    fig = None

    # Network tensors (all tensors end with _t to distinguish)
    learning_rate_t = None
    lr_son_t = None
    hr_father_t = None
    filters_t = None
    layers_t = None
    net_output_t = None
    loss_t = None
    train_op = None
    init_op = None

    # Parameters related to plotting and graphics
    plots = None
    loss_plot_space = None
    lr_son_image_space = None
    hr_father_image_space = None
    out_image_space = None


    def __init__(self, input_img, conf=Config(), ground_truth=None, kernels=None):
        # Acquire meta parameters configuration from configuration class as a class variable
        self.conf = conf
        self.cuda = conf.cuda
        # Read input image (can be either a numpy array or a path to an image file)
        self.input = input_img if type(input_img) is not str else img.imread(input_img)
        self.Y = False
        if len(self.input)==2:
            self.Y = True
        #input is ndarray
        # For evaluation purposes, ground-truth image can be supplied.
        self.gt = ground_truth if type(ground_truth) is not str else img.imread(ground_truth)
        #gt is ndarray
        # Preprocess the kernels. (see function to see what in includes).
        self.kernels = preprocess_kernels(kernels, conf)
        #downsample kernel custom
        # Prepare TF default computational graph
        # declare model here severs as initial model
        print(self.Y)
        self.model = simpleNet(self.Y)

        # Build network computational graph
        #self.build_network(conf)

        # Initialize network weights and meta parameters
        self.init_parameters()

        # The first hr father source is the input (source goes through augmentation to become a father)
        # Later on, if we use gradual sr increments, results for intermediate scales will be added as sources.
        self.hr_fathers_sources = [self.input]

        # We keep the input file name to save the output with a similar name. If array was given rather than path
        # then we use default provided by the configs
        self.file_name = input_img if type(input_img) is str else conf.name

    def run(self):
        # Run gradually on all scale factors (if only one jump then this loop only happens once)
        for self.sf_ind, (sf, self.kernel) in enumerate(zip(self.conf.scale_factors, self.kernels)):
            # verbose
            print('** Start training for sf=', sf, ' **')

            # Relative_sf (used when base change is enabled. this is when input is the output of some previous scale)
            # safe
            if np.isscalar(sf):
                sf = [sf, sf]
            self.sf = np.array(sf) / np.array(self.base_sf)
            self.output_shape = np.uint(np.ceil(np.array(self.input.shape[0:2]) * sf))
            print('input shape',self.input.shape)
            # Initialize network
            # reinit all for each scale factors, each gradual level
            self.init_parameters()
            if self.conf.init_net_for_each_sf == True:
                self.model = simpleNet(self.Y)
            if self.cuda:
                self.model = self.model.cuda()

            # Train the network
            # should be modified
            self.train()

            # Use augmented outputs and back projection to enhance result. Also save the result.
            post_processed_output = self.final_test()

            # Keep the results for the next scale factors SR to use as dataset
            self.hr_fathers_sources.append(post_processed_output)

            # In some cases, the current output becomes the new input. If indicated and if this is the right scale to
            # become the new base input. all of these conditions are checked inside the function.
            self.base_change()

            # Save the final output if indicated
            if self.conf.save_results:
                sf_str = ''.join('X%.2f' % s for s in self.conf.scale_factors[self.sf_ind])
                plt.imsave('%s/%s_zssr_%s.png' %
                           (self.conf.result_path, os.path.basename(self.file_name)[:-4], sf_str),
                           post_processed_output, vmin=0, vmax=1)

            # verbose
            print('** Done training for sf=', sf, ' **')

        # Return the final post processed output.
        # noinspection PyUnboundLocalVariable
        return post_processed_output

  

    def init_parameters(self):
        # Sometimes we only want to initialize some meta-params but keep the weights as they were
        # no need to init weight, done as model declaration
        # Initialize all counters etc
        #no need to change. For record here
        self.loss = [None] * self.conf.max_iters 
        self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
        self.iter = 0
        self.learning_rate = self.conf.learning_rate
        self.learning_rate_change_iter_nums = [0]

        # Downscale ground-truth to the intermediate sf size (for gradual SR).
        # This only happens if there exists ground-truth and sf is not the last one (or too close to it).
        # We use imresize with both scale and output-size, see comment in forward_backward_pass.
        # noinspection PyTypeChecker
        # scale_factor[-1] means the ground truth
        # hence sf/conf.scale_factors[-1] = 1.5/2, the target size for gt.
        self.gt_per_sf = (imresize(self.gt,
                                   scale_factor=self.sf / self.conf.scale_factors[-1],
                                   output_shape=self.output_shape,
                                   kernel=self.conf.downscale_gt_method)
                          if (self.gt is not None and
                              self.sf is not None and
                              np.any(np.abs(self.sf - self.conf.scale_factors[-1]) > 0.01))
                          else self.gt)

    def forward_backward_pass(self, lr_son, hr_father,criterion,optimizer):
        # First gate for the lr-son into the network is interpolation to the size of the father
        # Note: we specify both output_size and scale_factor. best explained by example: say father size is 9 and sf=2,
        # small_son size is 4. if we upscale by sf=2 we get wrong size, if we upscale to size 9 we get wrong sf.
        # The current imresize implementation supports specifying both.
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father.shape, self.conf.upscale_method)
        if self.Y == True:
            lr_son_input = torch.Tensor(interpolated_lr_son).unsqueeze_(0).unsqueeze_(0)
            hr_father = torch.Tensor(hr_father).unsqueeze_(0).unsqueeze_(0)
        else:
            lr_son_input = torch.Tensor(interpolated_lr_son).permute(2,0,1).unsqueeze_(0)
            hr_father = torch.Tensor(hr_father).permute(2,0,1).unsqueeze_(0)
        lr_son_input = lr_son_input.requires_grad_()
        
        if self.cuda == True:
            hr_father = hr_father.cuda()
            lr_son_input = lr_son_input.cuda()
      
        train_output = self.model(lr_son_input)
        loss = criterion(hr_father,train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.loss[self.iter] = loss

        return np.clip(np.squeeze(train_output.cpu().detach().numpy()), 0, 1)

    def forward_pass(self, lr_son, hr_father_shape=None):
        # First gate for the lr-son into the network is interpolation to the size of the father
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father_shape, self.conf.upscale_method)
        if self.Y == True:
            interpolated_lr_son = (torch.Tensor(interpolated_lr_son)).unsqueeze_(0).unsqueeze_(0)
        else:
            interpolated_lr_son = (torch.Tensor(interpolated_lr_son).permute(2,0,1)).unsqueeze_(0)
        if self.cuda:
            interpolated_lr_son = interpolated_lr_son.cuda()
        # Create feed dict
        

        # Run network
        return np.clip(np.squeeze(self.model(interpolated_lr_son).cpu().detach().permute(0,2,3,1).numpy()), 0, 1)

    def learning_rate_policy(self):
        # fit linear curve and check slope to determine whether to do nothing, reduce learning rate or finish
        if (not (1 + self.iter) % self.conf.learning_rate_policy_check_every
                and self.iter - self.learning_rate_change_iter_nums[-1] > self.conf.min_iters):
            # noinspection PyTupleAssignmentBalance
            #print(self.conf.run_test_every)
            [slope, _], [[var, _], _] = np.polyfit(self.mse_steps[-(self.conf.learning_rate_slope_range //
                                                                    self.conf.run_test_every):],
                                                   self.mse_rec[-(self.conf.learning_rate_slope_range //
                                                                  self.conf.run_test_every):],
                                                   1, cov=True)

            # We take the the standard deviation as a measure
            std = np.sqrt(var)

            # Verbose
            print('slope: ', slope, 'STD: ', std)

            # Determine learning rate maintaining or reduction by the ration between slope and noise
            if -self.conf.learning_rate_change_ratio * slope < std:
                self.learning_rate /= 10
                print("learning rate updated: ", self.learning_rate)

                # Keep track of learning rate changes for plotting purposes
                self.learning_rate_change_iter_nums.append(self.iter)

    def quick_test(self):
        # There are four evaluations needed to be calculated:

        # 1. True MSE (only if ground-truth was given), note: this error is before post-processing.
        # Run net on the input to get the output super-resolution (almost final result, only post-processing needed)
        self.sr = self.forward_pass(self.input)
        self.mse = (self.mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - self.sr)))]
                    if self.gt_per_sf is not None else None)

        # 2. Reconstruction MSE, run for reconstruction- try to reconstruct the input from a downscaled version of it
        self.reconstruct_output = self.forward_pass(self.father_to_son(self.input), self.input.shape)
        self.mse_rec.append(np.mean(np.ndarray.flatten(np.square(self.input - self.reconstruct_output))))

        # 3. True MSE of simple interpolation for reference (only if ground-truth was given)
        interp_sr = imresize(self.input, self.sf, self.output_shape, self.conf.upscale_method)
        self.interp_mse = (self.interp_mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - interp_sr)))]
                           if self.gt_per_sf is not None else None)

        # 4. Reconstruction MSE of simple interpolation over downscaled input
        interp_rec = imresize(self.father_to_son(self.input), self.sf, self.input.shape[0:2], self.conf.upscale_method)
        self.interp_rec_mse.append(np.mean(np.ndarray.flatten(np.square(self.input - interp_rec))))

        # Track the iters in which tests are made for the graphics x axis
        self.mse_steps.append(self.iter)

        # Display test results if indicated
        if self.conf.display_test_results:
            print('iteration: ', self.iter, 'reconstruct mse:', self.mse_rec[-1], ', true mse:', (self.mse[-1]
                                                                                                  if self.mse else None))

        # plot losses if needed
        if self.conf.plot_losses:
            self.plot()

    def train(self):
        #def loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr = self.learning_rate)
        # main training loop
        for self.iter in range(self.conf.max_iters):
            # Use augmentation from original input image to create current father.
            # If other scale factors were applied before, their result is also used (hr_fathers_in)
            self.hr_father = random_augment(ims=self.hr_fathers_sources,
                                            base_scales=[1.0] + self.conf.scale_factors,
                                            leave_as_is_probability=self.conf.augment_leave_as_is_probability,
                                            no_interpolate_probability=self.conf.augment_no_interpolate_probability,
                                            min_scale=self.conf.augment_min_scale,
                                            max_scale=([1.0] + self.conf.scale_factors)[len(self.hr_fathers_sources)-1],
                                            allow_rotation=self.conf.augment_allow_rotation,
                                            scale_diff_sigma=self.conf.augment_scale_diff_sigma,
                                            shear_sigma=self.conf.augment_shear_sigma,
                                            crop_size=self.conf.crop_size)

            # Get lr-son from hr-father
            self.lr_son = self.father_to_son(self.hr_father)
            #should convert input and output to torch tensor

            
            # run network forward and back propagation, one iteration (This is the heart of the training)
            self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father,criterion,optimizer)

            # Display info and save weights
            if not self.iter % self.conf.display_every:
                print('sf:', self.sf*self.base_sf, ', iteration: ', self.iter, ', loss: ', self.loss[self.iter])

            # Test network
            if self.conf.run_test and (not self.iter % self.conf.run_test_every):
                self.quick_test()

            # Consider changing learning rate or stop according to iteration number and losses slope
            self.learning_rate_policy()

            # stop when minimum learning rate was passed
            if self.learning_rate < self.conf.min_learning_rate:
                break

    def father_to_son(self, hr_father):
        # Create son out of the father by downscaling and if indicated adding noise
        lr_son = imresize(hr_father, 1.0 / self.sf, kernel=self.kernel)
        return np.clip(lr_son + np.random.randn(*lr_son.shape) * self.conf.noise_std, 0, 1)

    def final_test(self):
        # Run over 8 augmentations of input - 4 rotations and mirror (geometric self ensemble)
        outputs = []

        # The weird range means we only do it once if output_flip is disabled
        # We need to check if scale factor is symmetric to all dimensions, if not we will do 180 jumps rather than 90
        for k in range(0, 1 + 7 * self.conf.output_flip, 1 + int(self.sf[0] != self.sf[1])):
            # Rotate 90*k degrees and mirror flip when k>=4
            test_input = np.rot90(self.input, k) if k < 4 else np.fliplr(np.rot90(self.input, k))

            # Apply network on the rotated input
            tmp_output = self.forward_pass(test_input)

            # Undo the rotation for the processed output (mind the opposite order of the flip and the rotation)
            tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)

            # fix SR output with back projection technique for each augmentation
            for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                tmp_output = back_projection(tmp_output, self.input, down_kernel=self.kernel,
                                             up_kernel=self.conf.upscale_method, sf=self.sf)

            # save outputs from all augmentations
            outputs.append(tmp_output)

        # Take the median over all 8 outputs
        almost_final_sr = np.median(outputs, 0)

        # Again back projection for the final fused result
        for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
            almost_final_sr = back_projection(almost_final_sr, self.input, down_kernel=self.kernel,
                                              up_kernel=self.conf.upscale_method, sf=self.sf)

        # Now we can keep the final result (in grayscale case, colors still need to be added, but we don't care
        # because it is done before saving and for every other purpose we use this result)
        self.final_sr = almost_final_sr

        # Add colors to result image in case net was activated only on grayscale
        return self.final_sr

    def base_change(self):
        # If there is no base scale large than the current one get out of here
        if len(self.conf.base_change_sfs) < self.base_ind + 1:
            return

        # Change base input image if required (this means current output becomes the new input)
        if abs(self.conf.scale_factors[self.sf_ind] - self.conf.base_change_sfs[self.base_ind]) < 0.001:
            if len(self.conf.base_change_sfs) > self.base_ind:

                # The new input is the current output
                self.input = self.final_sr

                # The new base scale_factor
                self.base_sf = self.conf.base_change_sfs[self.base_ind]

                # Keeping track- this is the index inside the base scales list (provided in the config)
                self.base_ind += 1

            print('base changed to %.2f' % self.base_sf)

    def plot(self):
        plots_data, labels = zip(*[(np.array(x), l) for (x, l)
                                   in zip([self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse],
                                          ['True MSE', 'Reconstruct MSE', 'Bicubic to ground truth MSE',
                                           'Bicubic to reconstruct MSE']) if x is not None])

        # For the first iteration create the figure
        if not self.iter:
            # Create figure and split it using GridSpec. Name each region as needed
            self.fig = plt.figure(figsize=(9.5, 9))
            grid = GridSpec(4, 4)
            self.loss_plot_space = plt.subplot(grid[:-1, :])
            self.lr_son_image_space = plt.subplot(grid[3, 0])
            self.hr_father_image_space = plt.subplot(grid[3, 3])
            self.out_image_space = plt.subplot(grid[3, 1])

            # Activate interactive mode for live plot updating
            plt.ion()

            # Set some parameters for the plots
            self.loss_plot_space.set_xlabel('step')
            self.loss_plot_space.set_ylabel('MSE')
            self.loss_plot_space.grid(True)
            self.loss_plot_space.set_yscale('log')
            self.loss_plot_space.legend()
            self.plots = [None] * 4

            # loop over all needed plot types. if some data is none than skip, if some data is one value tile it
            self.plots = self.loss_plot_space.plot(*[[0]] * 2 * len(plots_data))

        # Update plots
        for plot, plot_data in zip(self.plots, plots_data):
            plot.set_data(self.mse_steps, plot_data)

            self.loss_plot_space.set_xlim([0, self.iter + 1])
            all_losses = np.array(plots_data)
            self.loss_plot_space.set_ylim([np.min(all_losses)*0.9, np.max(all_losses)*1.1])

        # Mark learning rate changes
        for iter_num in self.learning_rate_change_iter_nums:
            self.loss_plot_space.axvline(iter_num)

        # Add legend to graphics
        self.loss_plot_space.legend(labels)

        # Show current input and output images
        self.lr_son_image_space.imshow(self.lr_son, vmin=0.0, vmax=1.0)
        self.out_image_space.imshow(self.train_output, vmin=0.0, vmax=1.0)
        self.hr_father_image_space.imshow(self.hr_father, vmin=0.0, vmax=1.0)

        # These line are needed in order to see the graphics at real time
        self.fig.canvas.draw()
        plt.pause(0.01)
