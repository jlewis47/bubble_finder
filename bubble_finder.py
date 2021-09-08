import numpy as np
import matplotlib.pyplot as plt
import itertools

#####################################
####Functions for finding bubbles####
#####################################

def cycle_prev_pos(xtgt, ytgt, last_five_x, last_five_y):

    #print(last_five_x, xtgt)

    for i in range(len(last_five_y)-1):
        last_five_y[i]=last_five_y[i+1]
        last_five_x[i]=last_five_x[i+1]

    last_five_x[-1]=xtgt
    last_five_y[-1]=ytgt

    #print(last_five_x)
    #print('**************************************************')

    return(last_five_x, last_five_y)

def check_coord(xs,dim):
    """
    xs can be a single value OR a numpy array of values (-> no lists of tuples)

    deals with box periodicity
    If we go over the top of dim (int), xs goes over past 0
    If we go under 0, xs goes over past the top
    """
    return(xs%dim)

def check_next_step_2D(numbered_img, xtgt, ytgt, x_options, y_options, dim, obj_count):
        """
        Return whats needed to decide where to go
        Used when exploring bubble edges

        numbered_img is the 2D array we're processing
        xtgt and ytgt are the integer coordiantes of the cell where we are currently
        x_options and y_options are the coordinates of the cells we could go to with one cell step i
n any direction
        dim is the shape of the image
        obj_count is the ID of the object we're trying to map/fill

        returns:
        -nb_edges, score for each possible future cell highest score is the best move. Computed usin
g the number of edges (0s or neutral cells) that surround a given possible cell to move to, AND the 
number of potential future neighbouring cells that have already been flagged
        -valid_next_steps, boolean array that gives which steps are allowed (which neighbour cells a
re of value 1 -> neither neutral nor already visited) 
        """
        #find edges for next step
        nb_edges=np.zeros(len(x_options), dtype=np.int16)
        valid_next_steps=np.zeros(len(x_options), dtype=bool)
        
        x_nexts = check_coord(xtgt + x_options, dim[0])
        y_nexts = check_coord(ytgt + y_options, dim[1])
               
            
            
        valid_next_steps = numbered_img[y_nexts, x_nexts] == 1
        
        #print(valid_next_steps)
        #print(check_coord(xtgt, dim[0]), check_coord(ytgt, dim[1]), np.flipud(np.transpose(numbered
_img[y_nexts, x_nexts].reshape((3,3)))))

        x_next_options = check_coord(x_nexts + x_options[:,np.newaxis], dim[0])
        y_next_options = check_coord(y_nexts + y_options[:,np.newaxis], dim[1])    

        #number of neighbour edges (=0 cells) - f*number of neighbour =obj_count cells
        #the f factor probably has some fancy clever explanation, but was determined VERY empiricall
y -- it works so w/e but is it necessary?
        nb_edges = np.sum(numbered_img[y_next_options[:,:],x_next_options[:,:]] == 0, axis=1)# - 0.0
75*np.sum(numbered_img[y_next_options[:,:],x_next_options[:,:]] == obj_count, axis=1)
            
        return(nb_edges, valid_next_steps)

def sweep_grid_fill(numbered_img, obj_count, dim):
            """
            for an entry 2D image: 
            -find the smallest area containing obj_count values    
            -fill area between obj_count boundaries with obj_count, whilst leaving outside areas at 
0
    
            dim is the 2d shape of the image (array/tuple)
            """

            labeled_y, labeled_x = np.where(numbered_img==obj_count)
            #print(np.count_nonzero(numbered_img==1))

            obj_xmax=np.max(labeled_x)+1
            obj_ymax=np.max(labeled_y)+1
            obj_xmin=np.min(labeled_x)-1
            obj_ymin=np.min(labeled_y)-1
            
            loc_edge_copy = np.copy(numbered_img) #useful for tests as 1s are preserved (don't becom
e obj_count)

                
            #sweep up
            for ix in range(obj_xmin,obj_xmax):              
                
                writing=False                   
                
                ycoords = check_coord(np.arange(obj_ymin,obj_ymax),dim[1])
                
                borders_in_row = numbered_img[ycoords, ix]==obj_count
                if np.count_nonzero(borders_in_row)<1:continue                   
                
                ycoords_m1 = check_coord(ycoords-1, dim[1])

                cur_vals=numbered_img[ycoords, ix]
                #prev_vals=numbered_img[ycoords_m1, ix]        
        
                for iy in range(len(ycoords)):
                            
                    #to_void = cur_vals[iy]==0 and prev_vals[iy]==obj_count
                    #to_void = cur_vals[iy]==0 and loc_edge_copy[ycoords_m1[iy], ix]==obj_count
                    if cur_vals[iy]==0 and loc_edge_copy[ycoords_m1[iy], ix]==obj_count :
                        
                        writing = False
                    elif writing==False :
                        #from_void = cur_vals[iy]==1 and prev_vals[iy]==obj_count 
                        #from_void = cur_vals[iy]==1 and numbered_img[ycoords_m1[iy], ix]==obj_count
                        if cur_vals[iy]==1 and numbered_img[ycoords_m1[iy], ix]==obj_count : writing
=True
                
                    if writing and cur_vals[iy]==1: numbered_img[ycoords[iy], ix] = obj_count
                    

            #sweep right
            for ix in range(obj_xmin,obj_xmax):              
                
                writing=False                   
                
                ycoords = check_coord(np.arange(obj_ymin,obj_ymax),dim[1])[::-1]
                
                borders_in_row = numbered_img[ycoords, ix]==obj_count
                if np.count_nonzero(borders_in_row)<1:continue                   
                
                ycoords_p1 = check_coord(ycoords+1, dim[1])

                cur_vals=numbered_img[ycoords, ix]
                #prev_vals=numbered_img[ycoords_m1, ix]        
        
                for iy in range(len(ycoords)):
                            
                    #to_void = cur_vals[iy]==0 and prev_vals[iy]==obj_count
                    #to_void = cur_vals[iy]==0 and loc_edge_copy[ycoords_m1[iy], ix]==obj_count
                    if cur_vals[iy]==0 and loc_edge_copy[ycoords_p1[iy], ix]==obj_count :
                        
                        writing = False
                    elif writing==False :
                        #from_void = cur_vals[iy]==1 and prev_vals[iy]==obj_count 
                        #from_void = cur_vals[iy]==1 and numbered_img[ycoords_m1[iy], ix]==obj_count
                        if cur_vals[iy]==1 and numbered_img[ycoords_p1[iy], ix]==obj_count : writing
=True
                
                    if writing and cur_vals[iy]==1: numbered_img[ycoords[iy], ix] = obj_count
                    
                                                
                        
                        
                        
            # #sweep left            
            # for iy in range(obj_ymin,obj_ymax):
                            
            #     writing=False 

            #     xcoords=check_coord(np.arange(obj_xmin,obj_xmax), dim[0])

            #     borders_in_row = numbered_img[iy, xcoords]==obj_count
            #     if np.count_nonzero(borders_in_row)<1:continue                
                
            #     xcoords_m1=check_coord(xcoords-1, dim[0])                

            #     cur_vals=numbered_img[iy, xcoords]
            #     #prev_vals=numbered_img[iy, xcoords_m1]
            

            
            #     for ix in range(-obj_xmin+obj_xmax):               
                
            #         #to_void = cur_vals[ix]==0 and loc_edge_copy[iy, xcoords_m1[ix]]==obj_count
            #         if cur_vals[ix]==0 and loc_edge_copy[iy, xcoords_m1[ix]]==obj_count :
                        
            #             writing = False
            #         elif writing==False :
            #             #from_void = cur_vals[ix]==1 and numbered_img[iy, xcoords_m1[ix]]==obj_cou
nt                   
            #             if cur_vals[ix]==1 and numbered_img[iy, xcoords_m1[ix]]==obj_count : writi
ng=True
                        
            #         if writing and cur_vals[ix]==1: numbered_img[iy, xcoords[ix]] = obj_count 
                        

def explore_and_mark_2D(img, obj_count0=2):
    """
    return a copy of the boolean 2D array img,
    where 0s are left untouched, and contiguous regions of 1s are grouped together and receive uniqu
e integer IDs
   obj_count is the integer ID of the first object we shall ID. Due to the code's design, you must p
ick obj_count>=2
    """
    

    assert obj_count0>=2, 'Please pick obj_count >=2, lower values are kept reserved for proper code
 functionning' 

    dim=np.shape(img)
    
    x_options, y_options = np.array(list(itertools.product([-1,0,1],[-1,0,1]))).T
    
    numbered_img=np.copy(img)
    
    #find non zero pixels
    y_tgts,x_tgts=np.where(img==1)
    
    obj_count=obj_count0 #intial object count >2!!!
    

    xtgt=x_tgts[0]
    ytgt=y_tgts[0]    
    
    #print(xtgt,ytgt, numbered_img[ytgt, xtgt])    
    
    max_repeats=250 #if we run out of places to go, we go back for solutions
    #up to max_repeat times
    #warning this happens for EVERY OBJECT -- very large numbers cause large perf hits

    nsteps_max=20000 #hard limit to allow proceeding dispite code locking up -- tried to avoid this 
in testing but it is possible. May need ot adjust this based on grid size etc
    #in tests the number seemed to float around 10k steps

    last_x=np.zeros(max_repeats,dtype=np.int16) #watch out a big box could overstep the max size of 
16bit ints ?
    last_y=np.zeros(max_repeats,dtype=np.int16)

    
    istep=0
    
    new_obj=True
    
    repeat_counter=1
    used_rand_inds=[]

    # fig=plt.figure(1,figsize=(10,10))            
    # ax=fig.add_subplot(111)
    # plt.show(block=False)

    #MAIN LOOP
    while len(x_tgts)>25 and istep<nsteps_max:#allow for some margin of error (missed cells)

        plot=False

        # #catch weird overflow here
        # negs = numbered_img<0
        # if np.any(negs):numbered_img[negs]=1

        #print(istep, xtgt, ytgt)
        
        if new_obj: #if we just finished filling a bubble, then move to the next cell = 1 that we ca
n find to start again
            xtgt=x_tgts[0]
            ytgt=y_tgts[0]           

            last_x[:]=0
            last_y[:]=0

            last_x[-1]=xtgt
            last_y[-1]=ytgt

            new_obj=False
            
        nb_edges=np.zeros(len(x_options))
    
        if(istep==0): #first found =1 cell might have no edges ... increment x until we can start so
mewhere nice
    
    
            while np.all(nb_edges<=0):
    

    
                xtgt+=1
                nb_edges, valid_next_steps = check_next_step_2D(numbered_img, xtgt, ytgt, x_options,
 y_options, dim, obj_count)
            
                #print(xtgt, nb_edges, np.all(nb_edges==0))        
        
        else:
            
            nb_edges, valid_next_steps = check_next_step_2D(numbered_img, xtgt, ytgt, x_options, y_o
ptions, dim, obj_count)            
        
        #sort via nb of edges
        order = np.argsort(nb_edges)[::-1]
           

        
        if np.sum(nb_edges[valid_next_steps])>0: #if we have somewhere to go ... go there and change
 value

            #print('advance')

            #print(ytgt, xtgt)
            numbered_img[check_coord(ytgt,dim[1]),check_coord(xtgt,dim[0])]=obj_count             

            
            #first valid step that has the most edges
            xtgt = xtgt + x_options[order][valid_next_steps[order]][0]
            ytgt = ytgt + y_options[order][valid_next_steps[order]][0]       
        
            numbered_img[check_coord(ytgt,dim[1]),check_coord(xtgt,dim[0])]=obj_count             

            last_x, last_y = cycle_prev_pos(xtgt, ytgt, last_x, last_y)
            

        #if we don't have somewhere to go, we must either go back because there was a mistake...
        #or we assume we've just finished mapping the boudary of a bubble, and there are no possible
 paths to a cell bordering 0s ... So we then fill the created boudary
        
        elif repeat_counter<=max_repeats and np.sum(last_x!=0)>repeat_counter: #before we switch obj
ects we try going backwards once
            
            #print('repeat')

            #print(xtgt, ytgt)

            ytgt, xtgt = last_y[-repeat_counter], last_x[-repeat_counter]

            repeat_counter+=1
            
        else: #fill and switch objects
                
            #print('fill')

            if np.count_nonzero(numbered_img==obj_count)>3 : #Need something to fill in, 4 is the sm
allest size possible
                sweep_grid_fill(numbered_img, obj_count, dim)
            
            plot=True
            
            obj_count+=1

            repeat_counter=1#reset for new object

            y_tgts,x_tgts=np.where(numbered_img==1)
            
            new_obj=True            
            
        

        ####################
        ##diagnostic plots##
        ####################

        # if istep%250==0 or plot:
        
        #     plotting_img=np.zeros((dim[0]*2,dim[1]*2))
            
        #     plotting_img[:dim[0],:dim[1]]=numbered_img
        #     plotting_img[dim[0]:,dim[1]:]=numbered_img
        #     plotting_img[:dim[0],dim[1]:]=numbered_img
        #     plotting_img[dim[0]:,:dim[1]]=numbered_img
        

        #     img=ax.imshow(numbered_img)
        #     #plt.colorbar(img)
        #     #ax=plt.gca()
        #     #ax.axis("off")
        #     ax.plot(check_coord(last_x[-2], dim[0]),check_coord(last_y[-2], dim[1]),color='m', mar
ker='+', mew=1, ms=20)            
        #     ax.plot(check_coord(xtgt, dim[0]),check_coord(ytgt, dim[1]),color='r', marker='+', mew
=1, ms=20)

            
        #     #ax.set_xlim(xtgt-25+dim[0], xtgt+25+dim[0])
        #     #ax.set_.ylim(ytgt-25+dim[1], xtgt+25+dim[1])
        
        #     plt.pause(0.01)
        #     plt.cla()

        
        
        istep+=1
    

    #print(istep)

    if istep==nsteps_max:print('WARNING : stopping search as max steps reached...')
    
    return(obj_count,numbered_img)        

#########################################################
####Functions for handling simulation box periodicity####
#########################################################

def get_27(pos1,pos2,pos_vects):
    """
    Returns all distances including pos_vects reflections
    check that pos_vects and pos are same units !!!!
    """
    
    return((np.linalg.norm(pos1-(np.asarray(pos_vects)+np.asarray(pos2)[:,np.newaxis]),axis=2,ord=2)
))

def get_ctr(poss, pos_vects, dim):
    """
    Find the objects barycentre, accounting for edge periodicity of the coordinates

    post_vects is a silly construction that's needed by the code
    poss is the matrix of the 2D coordinates of the cells in the object
    dim is a shape array
    """

    prec = 5 #in nb of cells
    shp = np.ones(np.shape(poss))
    if np.any([poss<prec*shp,poss>(dim[0]-prec)*shp]):
        
        ctr=[poss[0,0],poss[0,1]]
        
        all_dists = get_27(ctr,poss,pos_vects)
        min_arg = np.argmin(all_dists,axis=1)
        ctr = np.average(poss+np.asarray(pos_vects)[min_arg],axis=0)

    else:
        
        ctr = np.average(poss,axis=0)
        
    return(ctr)

def mean_wrap(xcoords, ycoords, dim):
    """
    Wrapper for get_ctr, which finds the mean centre coordinates of a cloud of points in 2D, acocunt
ing for repetitions along the edges of the simulation box
    """
   
    ran=[-dim[0], 0, dim[0]]
    pos_vects = [[i,j] for j in ran for i in ran]
    
    new_coords=get_ctr(np.transpose([xcoords, ycoords]), pos_vects, dim)

    
    
    return(new_coords[0], new_coords[1])

########################################################
###Functions for merger objects from different slices###
########################################################

def link_slice_objects(slices):
    """
    Takes a list of outputs from explore_and_mark_2D, outputs a list of lists: linked_obj
    each list of linked_obj contains objects that have been linked together
    """

    linked_obj=[]
    
    dim = np.shape(slices[0])
    
    for iimg,img in enumerate(slices) :
        
        #print('slice %i'%iimg)
        
        #detect objects
        objects=np.unique(img)
        objects=objects[objects>1]
        
        nb_objects=len(objects)
        
        for iobj, obj in enumerate(objects):
            
            #print('     object %i'%obj)
            
            if obj in np.ravel(linked_obj) : continue #already found this
            
            #find centre cell
            ycells, xcells = np.where(img==obj)
            yctr, xctr = np.int16(mean_wrap(ycells, xcells, dim))

            yctr = check_coord_single(yctr, dim[1])
            xctr = check_coord_single(xctr, dim[0])            
            
            #print(xctr, yctr)
            
            slice_nb=0            
            obj_present=slices[slice_nb][yctr, xctr]>1
            #print(obj_present)
            found_objs=[]
            
            while obj_present and slice_nb < len(slices):
                
                #print('z=%i'%slice_nb)
                
                #print(slices[slice_nb][yctr, xctr])
              
                found_objs.append(slices[slice_nb][yctr, xctr])
                slice_nb+=1
                if slice_nb < len(slices) : obj_present=slices[slice_nb][yctr, xctr]>1 
                
                #print(obj_present)
                
    
  
            #print(found_objs)
            linked_obj.append(found_objs)
            
    return(linked_obj)

def merge_linked_obj(img_stack,links) : 
    """    
    Takes the stack of images processed by explore_and_mark_2D and the output from link_slice_object
s.
    Returns a new stack of images where objects have been merged along the 3rd axis
    """
    linked_img_stack=np.copy(img_stack)
    
    for group in links:
        
        for obj in group[1:]:
            
            linked_img_stack[linked_img_stack==obj]=group[0]
            
    return(linked_img_stack)
