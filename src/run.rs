use std::ffi::CString;
use std::fmt::Debug;
use std::sync::Arc;

pub use ort_sys::ONNXTensorElementDataType;

use crate::{AllocatorType, IntoTensorElementType, MemoryInfo, MemType, ortsys, RunOptions};
use crate::error::assert_non_null_pointer;

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error(transparent)]
    OrtError(#[from] crate::Error),
    #[error("error msg: {0}")]
    Msg(String),
}

/// allow &[T] or &mut [T] or Vec<T> or Box<[T]> or Arc<[T]>
pub struct RustOwnerValue<Container> {
    ptr: *mut ort_sys::OrtValue,
    owner: Container,
    _memory_info: MemoryInfo,
}

impl<Container> Drop for RustOwnerValue<Container> {
    fn drop(&mut self) {
        ortsys![unsafe ReleaseValue(self.ptr)];
    }
}

impl<Container> RustOwnerValue<Container> {
    pub fn into_container(mut self) -> Container {
        ortsys![unsafe ReleaseValue(self.ptr)];
        let _memory_info = std::mem::replace(&mut self._memory_info, unsafe { std::mem::zeroed() });
        let owner = std::mem::replace(&mut self.owner, unsafe { std::mem::zeroed() });
        std::mem::forget(self);
        owner
    }
}

impl<Container, T> RustOwnerValue<Container>
    where
        Container: std::ops::Deref<Target=[T]>,
        T: IntoTensorElementType + Debug + Clone + 'static,
{
    pub fn new(shape: &[i64], data: Container) -> crate::Result<Self, RunError> {
        let len = shape.iter().fold(1, |a, b| a * b);
        if data.len() < len as usize {
            return Err(RunError::Msg(format!("data len should be >= target len: [{} >= {}]?", data.len(), len)));
        }
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
            ) -> crate::Error::CreateTensorWithData;
            nonNull(value_ptr)
        ];
        let mut is_tensor = 0;
        ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> crate::Error::FailedTensorCheck];
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
    pub fn new_mut(shape: &[i64], mut data: Container) -> crate::Result<Self, RunError> {
        let len = shape.iter().fold(1, |a, b| a * b);
        if data.len() < len as usize {
            return Err(RunError::Msg(format!("data len should be >= target len: [{} >= {}]?", data.len(), len)));
        }
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
            ) -> crate::Error::CreateTensorWithData;
            nonNull(value_ptr)
        ];
        let mut is_tensor = 0;
        ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> crate::Error::FailedTensorCheck];
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

pub fn get_type_size(type_: ONNXTensorElementDataType) -> Result<usize, &'static str> {
    let size = match type_ {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED => { 0 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => { 4 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => { 1 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => { 1 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => { 2 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => { 2 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => { 4 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => { 8 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => { return Err("unsupported ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING"); }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => { 1 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => { 2 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => { 8 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => { 4 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => { 8 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 => { 8 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 => { 16 }
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => { 2 }
    };
    Ok(size)
}

pub fn convert_to_onnx_el_type(i: i32) -> Result<ONNXTensorElementDataType, String> {
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 as i32;
    const ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: i32 = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 as i32;
    let t = match i {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128
        }
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
        }
        _ => { return Err(format!("unknown type: {i}")); }
    };
    Ok(t)
}

impl<'a> RustOwnerValue<&'a [u8]> {
    /// for shared memory
    pub fn with_any_type(shape: &[i64], data: &'a [u8], type_: ONNXTensorElementDataType) -> crate::Result<Self, RunError> {
        let size = get_type_size(type_).unwrap();
        let len = shape.iter().fold(1, |a, b| a * b) as usize * size;
        if data.len() < len {
            return Err(RunError::Msg(format!("data len should be >= target len: [{} >= {}]?", data.len(), len)));
        }
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
                type_,
                &mut value_ptr
            ) -> crate::Error::CreateTensorWithData;
            nonNull(value_ptr)
        ];
        let mut is_tensor = 0;
        ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> crate::Error::FailedTensorCheck];
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
    pub fn with_any_type_mut(shape: &[i64], data: &'a mut [u8], type_: ONNXTensorElementDataType) -> crate::Result<Self, RunError> {
        let size = get_type_size(type_).unwrap();
        let len = shape.iter().fold(1, |a, b| a * b) as usize * size;
        if data.len() < len {
            return Err(RunError::Msg(format!("data len should be >= target len: [{} >= {}]?", data.len(), len)));
        }
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
                type_,
                &mut value_ptr
            ) -> crate::Error::CreateTensorWithData;
            nonNull(value_ptr)
        ];
        let mut is_tensor = 0;
        ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> crate::Error::FailedTensorCheck];
        assert_eq!(is_tensor, 1);
        Ok(Self {
            ptr: value_ptr,
            owner: data,
            _memory_info: memory_info,
        })
    }
}

pub struct Names<Container> {
    ptrs: Vec<*const std::ffi::c_char>,
    names: Container,
}

impl<T, Container> Names<Container>
    where
        Container: std::ops::Deref<Target=[T]>,
        T: AsRef<std::ffi::CStr>,
{
    #[inline]
    pub fn new(names: Container) -> Self {
        let mut ptrs = Vec::with_capacity(names.len());
        for name in names.iter() {
            let name = name.as_ref();
            ptrs.push(name.as_ptr());
        }
        Self {
            ptrs,
            names,
        }
    }
    #[inline]
    pub fn as_ptr(&self) -> *const *const std::ffi::c_char {
        self.ptrs.as_ptr()
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.ptrs.len()
    }
}

impl<T: AsRef<str>> From<Vec<T>> for Names<Vec<CString>> {
    fn from(value: Vec<T>) -> Self {
        let mut ptrs = Vec::with_capacity(value.len());
        let mut names = Vec::with_capacity(value.len());
        for name in value {
            let name = CString::new(name.as_ref()).unwrap();
            ptrs.push(name.as_ptr());
            names.push(name);
        }
        Self {
            ptrs,
            names,
        }
    }
}

impl<'a, T: AsRef<str>> From<&'a [T]> for Names<Vec<CString>> {
    fn from(value: &'a [T]) -> Self {
        let mut ptrs = Vec::with_capacity(value.len());
        let mut names = Vec::with_capacity(value.len());
        for name in value {
            let a = name.as_ref();
            let name = CString::new(a).unwrap();
            ptrs.push(name.as_ptr());
            names.push(name);
        }
        Self {
            ptrs,
            names,
        }
    }
}

impl<'a, T: AsRef<str>, const N: usize> From<[T; N]> for Names<Vec<CString>> {
    fn from(value: [T; N]) -> Self {
        let mut ptrs = Vec::with_capacity(value.len());
        let mut names = Vec::with_capacity(value.len());
        for name in value {
            let a = name.as_ref();
            let name = CString::new(a).unwrap();
            ptrs.push(name.as_ptr());
            names.push(name);
        }
        Self {
            ptrs,
            names,
        }
    }
}

pub struct Values<Container> {
    ptrs: Vec<*mut ort_sys::OrtValue>,
    values: Vec<RustOwnerValue<Container>>,
}

impl<Container> Values<Container> {
    #[inline]
    pub fn new(values_: Vec<RustOwnerValue<Container>>) -> Self {
        Self::from(values_)
    }
}

impl<T, Container> std::ops::Index<usize> for Values<Container>
    where
        Container: std::ops::Deref<Target=[T]>,
        T: IntoTensorElementType + Debug + Clone + 'static,
{
    type Output = RustOwnerValue<Container>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<T, Container> std::ops::IndexMut<usize> for Values<Container>
    where
        Container: std::ops::DerefMut<Target=[T]>,
        T: IntoTensorElementType + Debug + Clone + 'static,
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<T, Container> Values<Container>
    where
        Container: std::ops::Deref<Target=[T]>,
        T: IntoTensorElementType + Debug + Clone + 'static,
{
    #[inline]
    pub fn len(&self) -> usize {
        self.ptrs.len()
    }
    #[inline]
    pub fn as_ptr(&self) -> *const *const ort_sys::OrtValue {
        self.ptrs.as_ptr() as _
    }
    #[inline]
    pub fn as_slice(&self) -> &[RustOwnerValue<Container>] {
        self.values.as_slice()
    }
}

impl<T, Container> Values<Container>
    where
        Container: std::ops::DerefMut<Target=[T]>,
        T: IntoTensorElementType + Debug + Clone + 'static, {
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut *mut ort_sys::OrtValue {
        self.ptrs.as_mut_ptr()
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [RustOwnerValue<Container>] {
        self.values.as_mut_slice()
    }
}

impl<Container> From<Vec<RustOwnerValue<Container>>> for Values<Container> {
    #[inline]
    fn from(values_: Vec<RustOwnerValue<Container>>) -> Self {
        let mut values = Vec::with_capacity(values_.len());
        let mut ptrs = Vec::with_capacity(values.len());
        for value in values_ {
            let ptr = value.ptr;
            ptrs.push(ptr);
            values.push(value);
        }
        Self {
            ptrs,
            values,
        }
    }
}

impl super::Session {
    pub fn run_with_io_ref<I, O, SI, SO, CIn, COut, CNamesIn, CNamesOut>(&self,
                                                                         input_names: &Names<CNamesIn>,
                                                                         inputs: &[RustOwnerValue<CIn>],
                                                                         output_names: &Names<CNamesOut>,
                                                                         outputs: &mut [RustOwnerValue<COut>],
                                                                         run_options: Option<Arc<RunOptions>>) -> crate::Result<()>
        where
            CIn: std::ops::Deref<Target=[I]>,
            COut: std::ops::DerefMut<Target=[O]>,
            CNamesIn: std::ops::Deref<Target=[SI]>,
            CNamesOut: std::ops::Deref<Target=[SO]>,
            I: IntoTensorElementType + Debug + Clone + 'static,
            O: IntoTensorElementType + Debug + Clone + 'static,
            SI: AsRef<std::ffi::CStr>,
            SO: AsRef<std::ffi::CStr>,
    {
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
				input_names.as_ptr(),
				input_ort_values.as_ptr(),
				input_ort_values.len() as _,
				output_names.as_ptr(),
				output_names.len() as _,
				output_tensor_ptrs.as_mut_ptr()
			) -> crate::Error::SessionRun
		];
        Ok(())
    }

    pub fn run_with_values<I, O, SI, SO, CIn, COut, CNamesIn, CNamesOut>(&self,
                                                                         input_names: &Names<CNamesIn>,
                                                                         inputs: &Values<CIn>,
                                                                         output_names: &Names<CNamesOut>,
                                                                         outputs: &mut Values<COut>,
                                                                         run_options: Option<Arc<RunOptions>>) -> crate::Result<()>
        where
            CIn: std::ops::Deref<Target=[I]>,
            COut: std::ops::DerefMut<Target=[O]>,
            CNamesIn: std::ops::Deref<Target=[SI]>,
            CNamesOut: std::ops::Deref<Target=[SO]>,
            I: IntoTensorElementType + Debug + Clone + 'static,
            O: IntoTensorElementType + Debug + Clone + 'static,
            SI: AsRef<std::ffi::CStr>,
            SO: AsRef<std::ffi::CStr>,
    {
        // The C API expects pointers for the arrays (pointers to C-arrays)
        let run_options_ptr = if let Some(run_options) = &run_options {
            run_options.run_options_ptr
        } else {
            std::ptr::null_mut()
        };
        ortsys![
			unsafe Run(
				self.inner.session_ptr,
				run_options_ptr,
				input_names.as_ptr(),
				inputs.as_ptr(),
				inputs.len() as _,
				output_names.as_ptr(),
				output_names.len() as _,
				outputs.as_mut_ptr()
			) -> crate::Error::SessionRun
		];
        Ok(())
    }
}