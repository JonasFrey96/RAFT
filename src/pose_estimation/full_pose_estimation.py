from .flow_to_trafo_PnP import flow_to_trafo_PnP
from loss import AddSLoss
import torch
def full_pose_estimation(
	h_gt, 
	h_render,
	h_init,
	bb, 
	flow_valid, 
	flow_pred, 
	flow_gt,
	idx,
	K_ren,
	K_real,
	render_d,
	model_points, 
	cfg = {}
	):
	"""
	
	Returns ADD-S for each element in batch!
	"""

	real_tl, real_br, ren_tl, ren_br = bb
	typ = flow_pred.dtype
	
	h_pred = h_init.clone()
	ratios = []

	sucs = []

	for b in range( real_tl.shape[0] ):
			
		suc, h_pre, ratio = flow_to_trafo_PnP( 
			real_br = real_br[b].type(typ).to('cpu'), 
			real_tl = real_tl[b].type(typ).to('cpu'), 
			ren_br = ren_br[b].type(typ).to('cpu'), 
			ren_tl = ren_tl[b].type(typ).to('cpu'), 
			flow_mask = (flow_valid[b][:,:].to('cpu') == 1), 
			u_map = flow_pred[b][ 0, :, :].to('cpu'), 
			v_map = flow_pred[b][ 1, :, :].to('cpu'), 
			K_ren = K_ren[b].type(typ).to('cpu'), 
			K_real = K_real[b].to('cpu'), 
			render_d = render_d[b].to('cpu'), 
			h_render = h_render[b].to('cpu').clone(),
			h_real_est = h_init[b].to('cpu').clone(),
			** cfg.get("flow_to_trafo_PnP", {}) #
			)
		sucs.append(suc)
		h_pred[b,:,:] = h_pre.clone().to(h_init.device)
		ratios.append(ratio)


	h_pred.to('cuda')
	h_gt = h_gt.type(torch.float32)
	target = torch.bmm ( model_points, torch.transpose(h_gt[:,:3,:3], 1,2 )  ) + h_gt[:,:3,3][:,None,:].repeat(1, model_points.shape[1],1)

	typ = torch.float32
	h_pred = h_pred.type(typ).to('cuda')
	h_gt = h_gt.type(typ).to('cuda')
	h_init = h_init.type(typ).to('cuda')
	
	adds = AddSLoss( sym_list = list( range(0,22)))
	add_s = AddSLoss( sym_list = [12,15,18,19,20] ) # bowl, wood_block, large clamp, extra_large clamp, foam_brick

	
	add_s_h_pred = add_s( target.clone(), model_points.clone(), idx, H=h_pred)
	# if ( adds_h_pred > 0.04).sum() > 0:
		# print("ratio", ratios, "adds_h_pred", adds_h_pred)
	add_s_h_init = add_s( target.clone(), model_points.clone(), idx, H=h_init) 
	add_s_h_gt = add_s( target.clone(), model_points.clone(), idx, H=h_gt)

	adds_h_pred = adds( target.clone(), model_points.clone(), idx, H=h_pred)
	# if ( adds_h_pred > 0.04).sum() > 0:
		# print("ratio", ratios, "adds_h_pred", adds_h_pred)
	adds_h_init = adds( target.clone(), model_points.clone(), idx, H=h_init) 
	adds_h_gt = adds( target.clone(), model_points.clone(), idx, H=h_gt)


	limit = torch.norm( h_pred[:,:3,3] - h_init[:,:3,3], dim=1, p=2) > 0.04	

	if limit.sum() == real_tl.shape[0] :
		print( "Failed NORM: ", torch.norm( h_pred[:,:3,3] - h_init[:,:3,3], dim=1, p=2), "ADDs" ,adds_h_pred  )
		return {}, limit.sum()
	elif (adds_h_pred>0.05).sum() > 0 :
		print( "Failed ADDS: ", adds_h_pred	)
		return {}, limit.sum()
	
	valid = (limit + (adds_h_pred>0.05)) == False
	count_invalid = real_tl.shape[0]- valid.sum()
	# if limit.sum() ==  real_tl.shape[0]:
	# 	return {}, limit.sum()


	adds_h_init = adds_h_init[valid]
	# adds_h_gt = adds_h_gt[limit==False]
	adds_h_pred = adds_h_pred[valid]
	# print( adds_h_pred* 100 , "cm ")
	add_s_h_init = add_s_h_init[valid]
	# add_s_h_gt = add_s_h_gt[limit==False]
	add_s_h_pred = add_s_h_pred[valid]
	# print("IDX min max", idx.min(), idx.max() )
	return {"adds_h_pred": adds_h_pred,
					"adds_h_init": adds_h_init,
					# "adds_h_gt": adds_h_gt,
					"add_s_h_pred": add_s_h_pred,
					"add_s_h_init": add_s_h_init,
					# "add_s_h_gt": add_s_h_gt
					}, count_invalid