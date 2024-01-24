use std::ffi::CString;
use std::fmt::Debug;
use std::os::raw::c_char;
use std::sync::Arc;
use ort_sys::ONNXTensorElementDataType;
use crate::{AllocatorType, Error, IntoTensorElementType, MemoryInfo, MemType, ortsys, RunOptions};
use crate::error::assert_non_null_pointer;

/// allow &[T] or &mut [T] or Vec<T> or Box<[T]> or Arc<[T]>
pub struct RustOwnerValue<Container> {
    ptr: *mut ort_sys::OrtValue,
    owner: Container,
    _memory_info: MemoryInfo,
}

impl<Container, T> RustOwnerValue<Container>
    where
        Container: std::ops::Deref<Target=[T]>,
        T: IntoTensorElementType + Debug + Clone + 'static,
{
    pub fn new(shape: &[i64], data: Container) -> crate::Result<Self> {
        let len = shape.iter().fold(1, |a, b| a * b);
        assert_eq!(len as usize, data.len());
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = shape.len();
        let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemType::Default)?;
        let tensor_values_ptr: *mut std::ffi::c_void = data.as_ptr() as *mut std::ffi::c_void;
        assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;
        let mut value_ptr: *mut ort_sys::OrtValue = std::ptr::null_mut();
        ortsys![
            unsafe CreateTensorWithDataAsOrtValue(
                memory_info.ptr,
                tensor_values_ptr,
                (data.len() * std::mem::size_of::<T>()) as _,
                shape_ptr,
                shape_len as _,
                T::into_tensor_element_type().into(),
                &mut value_ptr
            ) -> Error::CreateTensorWithData;
            nonNull(value_ptr)
        ];
        let mut is_tensor = 0;
        ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> Error::FailedTensorCheck];
        assert_eq!(is_tensor, 1);
        Ok(Self {
            ptr: value_ptr,
            owner: data,
            _memory_info: memory_info,
        })
    }
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &*self.owner
    }

    #[inline]
    pub fn ptr(&self) -> *const ort_sys::OrtValue {
        self.ptr as _
    }
}

impl<Container, T> RustOwnerValue<Container>
    where
        Container: std::ops::DerefMut<Target=[T]>,
        T: IntoTensorElementType + Debug + Clone + 'static,
{
    pub fn new_mut(shape: &[i64], mut data: Container) -> crate::Result<Self> {
        let len = shape.iter().fold(1, |a, b| a * b);
        assert_eq!(len as usize, data.len());
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = shape.len();
        let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemType::Default)?;
        let tensor_values_ptr: *mut std::ffi::c_void = data.as_mut_ptr() as *mut std::ffi::c_void;
        assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;
        let mut value_ptr: *mut ort_sys::OrtValue = std::ptr::null_mut();
        ortsys![
            unsafe CreateTensorWithDataAsOrtValue(
                memory_info.ptr,
                tensor_values_ptr,
                (data.len() * std::mem::size_of::<T>()) as _,
                shape_ptr,
                shape_len as _,
                T::into_tensor_element_type().into(),
                &mut value_ptr
            ) -> Error::CreateTensorWithData;
            nonNull(value_ptr)
        ];
        let mut is_tensor = 0;
        ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> Error::FailedTensorCheck];
        assert_eq!(is_tensor, 1);
        Ok(Self {
            ptr: value_ptr,
            owner: data,
            _memory_info: memory_info,
        })
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut *self.owner
    }

    #[inline]
    pub fn ptr_mut(&mut self) -> *mut ort_sys::OrtValue {
        self.ptr
    }
}

impl<'a> RustOwnerValue<&'a [u8]> {
    /// for shared memory
    pub fn with_any_type(shape: &[i64], data: &'a [u8], type_: i32) -> crate::Result<Self> {
        assert!(type_ >= ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED as i32 &&
            type_ <= ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 as i32);
        let len = shape.iter().fold(1, |a, b| a * b);
        assert_eq!(len as usize, data.len());
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = shape.len();
        let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemType::Default)?;
        let tensor_values_ptr: *mut std::ffi::c_void = data.as_ptr() as *mut std::ffi::c_void;
        assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;
        let mut value_ptr: *mut ort_sys::OrtValue = std::ptr::null_mut();
        ortsys![
            unsafe CreateTensorWithDataAsOrtValue(
                memory_info.ptr,
                tensor_values_ptr,
                data.len() as _,
                shape_ptr,
                shape_len as _,
                std::mem::transmute(type_),
                &mut value_ptr
            ) -> Error::CreateTensorWithData;
            nonNull(value_ptr)
        ];
        let mut is_tensor = 0;
        ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> Error::FailedTensorCheck];
        assert_eq!(is_tensor, 1);
        Ok(Self {
            ptr: value_ptr,
            owner: data,
            _memory_info: memory_info,
        })
    }
}

impl<'a> RustOwnerValue<&'a mut [u8]> {
    /// for shared memory
    pub fn with_any_type_mut(shape: &[i64], data: &'a mut [u8], type_: i32) -> crate::Result<Self> {
        assert!(type_ >= ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED as i32 &&
            type_ <= ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 as i32);
        let len = shape.iter().fold(1, |a, b| a * b);
        assert_eq!(len as usize, data.len());
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = shape.len();
        let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemType::Default)?;
        let tensor_values_ptr: *mut std::ffi::c_void = data.as_mut_ptr() as *mut std::ffi::c_void;
        assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;
        let mut value_ptr: *mut ort_sys::OrtValue = std::ptr::null_mut();
        ortsys![
            unsafe CreateTensorWithDataAsOrtValue(
                memory_info.ptr,
                tensor_values_ptr,
                data.len() as _,
                shape_ptr,
                shape_len as _,
                std::mem::transmute(type_),
                &mut value_ptr
            ) -> Error::CreateTensorWithData;
            nonNull(value_ptr)
        ];
        let mut is_tensor = 0;
        ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> Error::FailedTensorCheck];
        assert_eq!(is_tensor, 1);
        Ok(Self {
            ptr: value_ptr,
            owner: data,
            _memory_info: memory_info,
        })
    }
}

impl super::Session {
    pub fn run_io<I, O, CIn, COut>(&self,
                                   input_names: &[&str],
                                   inputs: &[RustOwnerValue<CIn>],
                                   outputs: &mut [RustOwnerValue<COut>],
                                   run_options: Option<Arc<RunOptions>>) -> crate::Result<()>
        where
            CIn: std::ops::Deref<Target=[I]>,
            COut: std::ops::DerefMut<Target=[O]>,
            I: IntoTensorElementType + Debug + Clone + 'static,
            O: IntoTensorElementType + Debug + Clone + 'static,
    {
        assert_eq!(outputs.len(), self.outputs.len());
        let input_names_ptr: Vec<*const c_char> = input_names
            .iter()
            .map(|n| CString::new(*n).unwrap())
            .map(|n| n.into_raw() as *const c_char)
            .collect();
        let output_names_ptr: Vec<*const c_char> = self
            .outputs
            .iter()
            .map(|output| CString::new(output.name.as_str()).unwrap())
            .map(|n| n.into_raw() as *const c_char)
            .collect();
        // The C API expects pointers for the arrays (pointers to C-arrays)
        let input_ort_values: Vec<*const ort_sys::OrtValue> = inputs.iter().map(|a| a.ptr()).collect();
        let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = outputs.iter_mut().map(|a| a.ptr_mut()).collect();
        let run_options_ptr = if let Some(run_options) = &run_options {
            run_options.run_options_ptr
        } else {
            std::ptr::null_mut()
        };
        ortsys![
			unsafe Run(
				self.inner.session_ptr,
				run_options_ptr,
				input_names_ptr.as_ptr(),
				input_ort_values.as_ptr(),
				input_ort_values.len() as _,
				output_names_ptr.as_ptr(),
				output_names_ptr.len() as _,
				output_tensor_ptrs.as_mut_ptr()
			) -> Error::SessionRun
		];
        // Reconvert name ptrs to CString so drop impl is called and memory is freed
        drop(
            input_names_ptr
                .into_iter()
                .chain(output_names_ptr.into_iter())
                .map(|p| {
                    assert_non_null_pointer(p, "c_char for CString")?;
                    unsafe { Ok(CString::from_raw(p as *mut c_char)) }
                })
                .collect::<crate::Result<Vec<_>>>()?
        );
        Ok(())
    }
}