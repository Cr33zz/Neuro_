/*
 * Copyright (c) 2018 vargaconsulting, Toronto,ON Canada
 * Author: Varga, Steven <steven@vargaconsulting.ca>
 *
 */

#include <hdf5.h>
#include "macros.h"
#include "utils.hpp"
#include <limits>
#include <initializer_list>

#ifndef  H5CPP_CREATE_H 
#define H5CPP_CREATE_H

namespace h5 { namespace impl {
	inline hid_t create(hid_t fd, const std::string& path_,
			size_t rank,  const hsize_t* max_dims,  const hsize_t* chunk_dims, int32_t deflate, hid_t type){

		hsize_t current_dims[H5CPP_MAX_RANK];
		hid_t group, dcpl,space, ds;

		std::pair<std::string, std::string> path = h5::utils::split_path( path_ );
		// creates path if doesn't exists otherwise opens it
		if(!path.first.empty()){
			group = h5::utils::group_exist(fd,path.first,true);
			if( group < 0)
				return group;
			}else
				group = fd;

		dcpl = H5Pcreate(H5P_DATASET_CREATE);
		// NaN is platform and type dependent, branch out on native types
		if( H5Tequal(H5T_NATIVE_DOUBLE, type) ){
			double value = std::numeric_limits<double>::quiet_NaN();
			H5Pset_fill_value( dcpl, type, &value  );
		}else if( H5Tequal(H5T_NATIVE_FLOAT, type) ){
			float value = std::numeric_limits<float>::quiet_NaN();
			H5Pset_fill_value( dcpl, type, &value );
		}else if( H5Tequal(H5T_NATIVE_LDOUBLE, type) ){
			long double value = std::numeric_limits<long double>::quiet_NaN();
			H5Pset_fill_value( dcpl, type, &value );
		}

		// this prevents unreadable datasets in hdf-view or julia
		H5Pset_fill_time(dcpl,H5D_FILL_TIME_ALLOC);
		// add support for chunks only if specified
		if( *chunk_dims ){
			// set current dimensions to given one or zero if H5S_UNLIMITED
			// this mimics matlab(tm) behavior while allowing extendable matrices
			for(hsize_t i=0;i<rank;i++)
				current_dims[i] = max_dims[i] != H5S_UNLIMITED ? max_dims[i] : static_cast<hsize_t>(0);

			H5Pset_chunk(dcpl, rank, chunk_dims);
			if( deflate ) H5Pset_deflate (dcpl, deflate);
		} else
			for(hsize_t i=0;i<rank;i++) current_dims[i] = max_dims[i];

		space = H5Screate_simple( rank, current_dims, max_dims );
		ds = H5Dcreate2(group, path.second.data(), type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
		if( !path.first.empty() )
			H5Gclose( group );
		H5Pclose(dcpl); H5Sclose( space); H5Tclose(type);
		return ds;
	}
}}



/* @namespace h5
 * @brief public namespace
 */
namespace h5 {

	/** \ingroup io-create 
	 * \brief **T** template parameter defines the underlying representation of dataset created within HDF5 filesystem 
	 * referenced by **fd** descriptor.
	 * **path** behaves much similar to POSIX files system path: either relative or absolute. HDF5 supports 
	 * arbitrary number of dimensions which is specified by **max_dim**, and **chunk_size** controls how this 
	 * array is accessed. When chunked access is used keep in mind small values may result in excessive 
	 * data cache operations.<br>
	 * This function is general, ca
	 *
	 * **TODO:**
	 * 		provide mechanism to specify attributes, filters, ...  
	 *
	 * @param fd opened HDF5 file descripor
	 * @param path full path where the newly created object will be placed
	 * @param max_dims size of the object, H5S_UNLIMITED to mark extendable dimension
	 * @param chunk_dims for better performance data sets maybe stored in chunks, which is a unit size 
	 * 		  for IO operations. Streaming, and filters may be applied only on chunked datasets.
	 * @param deflate 0-9 controls [GZIP][10] compression.
	 * @tparam T [unsigned](int8_t|int16_t|int32_t|int64_t) | (float|double)    
	 * @return opened dataset descriptor of hid_t, that must be closed with [H5Dclose][5]  
	 * \code
	 * example:
	 * 		hid_t ds = create<double>(fd, "matrix/double type",{100,10},{1,10}, 9); 	 
	 * 		hid_t ds = create<short>(fd, "array/short",{100,10,10});          			 
	 * 		hid_t ds = create<float>(fd, "stream",{H5S_UNLIMITED},{10}, 9); 			
	 *  																				
	 * 		hid_t ds = h5::create<float>(fd,"stl/partial",{100,vf.size()},{1,10}, 9);    
	 * 			h5::write(ds, vf, {2,0},{1,10} ); 										 	
	 * 			auto rvo  = h5::read< std::vector<float>>(ds); 							
	 * 		H5Dclose(ds); 																
	 * \endcode 
	 
	 
	 * @sa open close [gzip][10] [H5Fcreate][1] [H5Fopen][2] [H5Fclose][3] [H5Dopen][4] [H5Dclose][5] 
	 * [1]: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-Create
	 * [2]: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-Open
	 * [3]: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-Close
	 * [4]: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5D.html#Dataset-Open
	 * [5]: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5D.html#Dataset-Close
	 * [10]: https://support.hdfgroup.org/HDF5/Tutor/compress.html
	 */
	template <typename T> inline hid_t create(hid_t fd, const std::string& path,
			std::initializer_list<hsize_t> max_dims, std::initializer_list<hsize_t> chunk_dims={0},
			const int32_t deflate = H5CPP_NO_COMPRESSION ){

		return impl::create(fd,path,max_dims.size(), max_dims.begin(), chunk_dims.begin(), deflate, utils::h5type<T>() );
   	}

	/** \ingroup io-create 
	 * @brief create dataset within HDF5 file space with dimensions extracted from references object 
	 * @param fd opened HDF5 file descripor
	 * @param path full path where the newly created object will be placed
	 * @param ref stl|arma|eigen valid templated object with dimensions       
	 * @tparam T [unsigned](int8_t|int16_t|int32_t|int64_t)| (float|double)    
	 * @return opened dataset descriptor of hid_t, that must be closed with [H5Dclose][5] 
	 * \code
	 * example:
	 * 	#include <hdf5.h>
	 * 	#include <armadillo>
	 * 	#include <h5cpp/all>
	 *  
	 *  int main(){
	 * 		hid_t fd = h5::create("some_file.h5"); 		// create file
	 * 		arma::mat M(100,10); 						// define object
	 * 		hid_t ds = h5::create(fd,"matrix",M); 		// create dataset from object
	 * 		H5Dclose(ds); 								// close dataset
	 * 		h5::close(fd); 								// and file descriptor
	 * 	}
	 * \endcode
	 * @sa open close [gzip][10] [H5Fcreate][1] [H5Fopen][2] [H5Fclose][3] [H5Dopen][4] [H5Dclose][5] 
	 * [1]: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-Create
	 * [2]: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-Open
	 * [3]: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-Close
	 * [4]: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5D.html#Dataset-Open
	 * [5]: https://support.hdfgroup.org/HDF5/doc/RM/RM_H5D.html#Dataset-Close
	 * [10]: https://support.hdfgroup.org/HDF5/Tutor/compress.html
	 */
	template<typename T, typename BaseType = typename utils::base<T>::type, size_t Rank = utils::base<T>::rank >
		inline hid_t create(  hid_t fd, const std::string& path, const T& ref ){

		std::array<hsize_t,Rank> max_dims = h5::utils::get_dims( ref );
		std::array<hsize_t,Rank> chunk_dims={}; // initialize to zeros
 		return impl::create(fd,path,Rank,max_dims.data(),chunk_dims.data(),H5CPP_NO_COMPRESSION, utils::h5type<BaseType>());
	}
}
#endif

