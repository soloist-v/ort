use std::{ffi::CString, os::raw::c_char};

use super::{char_p_to_string, error::Result, ortfree, ortsys, Error};

/// Container for model metadata, including name & producer information.
pub struct ModelMetadata {
	metadata_ptr: *mut ort_sys::OrtModelMetadata,
	allocator_ptr: *mut ort_sys::OrtAllocator
}

impl ModelMetadata {
	pub(crate) fn new(metadata_ptr: *mut ort_sys::OrtModelMetadata, allocator_ptr: *mut ort_sys::OrtAllocator) -> Self {
		ModelMetadata { metadata_ptr, allocator_ptr }
	}

	/// Gets the model description, returning an error if no description is present.
	pub fn description(&self) -> Result<String> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		ortsys![unsafe ModelMetadataGetDescription(self.metadata_ptr, self.allocator_ptr, &mut str_bytes) -> Error::GetModelMetadata; nonNull(str_bytes)];

		let value = match char_p_to_string(str_bytes) {
			Ok(value) => value,
			Err(e) => {
				ortfree!(unsafe self.allocator_ptr, str_bytes);
				return Err(e);
			}
		};
		ortfree!(unsafe self.allocator_ptr, str_bytes);
		Ok(value)
	}

	/// Gets the model producer name, returning an error if no producer name is present.
	pub fn producer(&self) -> Result<String> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		ortsys![unsafe ModelMetadataGetProducerName(self.metadata_ptr, self.allocator_ptr, &mut str_bytes) -> Error::GetModelMetadata; nonNull(str_bytes)];

		let value = match char_p_to_string(str_bytes) {
			Ok(value) => value,
			Err(e) => {
				ortfree!(unsafe self.allocator_ptr, str_bytes);
				return Err(e);
			}
		};
		ortfree!(unsafe self.allocator_ptr, str_bytes);
		Ok(value)
	}

	/// Gets the model name, returning an error if no name is present.
	pub fn name(&self) -> Result<String> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		ortsys![unsafe ModelMetadataGetGraphName(self.metadata_ptr, self.allocator_ptr, &mut str_bytes) -> Error::GetModelMetadata; nonNull(str_bytes)];

		let value = match char_p_to_string(str_bytes) {
			Ok(value) => value,
			Err(e) => {
				ortfree!(unsafe self.allocator_ptr, str_bytes);
				return Err(e);
			}
		};
		ortfree!(unsafe self.allocator_ptr, str_bytes);
		Ok(value)
	}

	/// Gets the model version, returning an error if no version is present.
	pub fn version(&self) -> Result<i64> {
		let mut ver = 0i64;
		ortsys![unsafe ModelMetadataGetVersion(self.metadata_ptr, &mut ver) -> Error::GetModelMetadata];
		Ok(ver)
	}

	/// Fetch the value of a custom metadata key. Returns `Ok(None)` if the key is not found.
	pub fn custom(&self, key: &str) -> Result<Option<String>> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		let key_str = CString::new(key)?;
		ortsys![unsafe ModelMetadataLookupCustomMetadataMap(self.metadata_ptr, self.allocator_ptr, key_str.as_ptr(), &mut str_bytes) -> Error::GetModelMetadata];
		if !str_bytes.is_null() {
			let value = match char_p_to_string(str_bytes) {
				Ok(value) => value,
				Err(e) => {
					ortfree!(unsafe self.allocator_ptr, str_bytes);
					return Err(e);
				}
			};
			ortfree!(unsafe self.allocator_ptr, str_bytes);
			Ok(Some(value))
		} else {
			Ok(None)
		}
	}
}

impl Drop for ModelMetadata {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseModelMetadata(self.metadata_ptr)];
	}
}
