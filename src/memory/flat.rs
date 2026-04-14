use alloc::vec;
use alloc::vec::Vec;
use super::super::{Error, RISCV_PAGE_SHIFTS, RISCV_PAGESIZE, Register, error::OutOfBoundKind};
use super::{Memory, check_no_overflow, fill_page_data, get_page_indices, memset, set_dirty};

use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

pub struct FlatMemory<R> {
    data: Vec<u8>,
    flags: Vec<u8>,
    memory_size: usize,
    riscv_pages: usize,
    load_reservation_address: R,
    _inner: PhantomData<R>,
}

impl<R> Deref for FlatMemory<R> {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<R> DerefMut for FlatMemory<R> {
    fn deref_mut(&mut self) -> &mut Vec<u8> {
        &mut self.data
    }
}

/// A flat chunk of memory used for RISC-V machine, it lacks all the permission
/// checking logic.
impl<R: Register> Memory for FlatMemory<R> {
    type REG = R;

    fn new(memory_size: usize) -> Self {
        assert!(memory_size.is_multiple_of(RISCV_PAGESIZE));
        Self {
            data: vec![0; memory_size],
            flags: vec![0; memory_size / RISCV_PAGESIZE],
            memory_size,
            riscv_pages: memory_size / RISCV_PAGESIZE,
            load_reservation_address: R::from_u64(u64::MAX),
            _inner: PhantomData,
        }
    }

    fn init_pages(
        &mut self,
        addr: u64,
        size: u64,
        _flags: u8,
        source: Option<Bytes>,
        offset_from_addr: u64,
    ) -> Result<(), Error> {
        fill_page_data(self, addr, size, source, offset_from_addr)
    }

    fn fetch_flag(&mut self, page: u64) -> Result<u8, Error> {
        if page < self.riscv_pages as u64 {
            Ok(self.flags[page as usize])
        } else {
            Err(Error::MemOutOfBound(
                page << RISCV_PAGE_SHIFTS,
                OutOfBoundKind::Memory,
            ))
        }
    }

    fn set_flag(&mut self, page: u64, flag: u8) -> Result<(), Error> {
        if page < self.riscv_pages as u64 {
            self.flags[page as usize] |= flag;
            Ok(())
        } else {
            Err(Error::MemOutOfBound(
                page << RISCV_PAGE_SHIFTS,
                OutOfBoundKind::Memory,
            ))
        }
    }

    fn clear_flag(&mut self, page: u64, flag: u8) -> Result<(), Error> {
        if page < self.riscv_pages as u64 {
            self.flags[page as usize] &= !flag;
            Ok(())
        } else {
            Err(Error::MemOutOfBound(
                page << RISCV_PAGE_SHIFTS,
                OutOfBoundKind::Memory,
            ))
        }
    }

    fn memory_size(&self) -> usize {
        self.memory_size
    }

    fn execute_load16(&mut self, addr: u64) -> Result<u16, Error> {
        self.load16(&Self::REG::from_u64(addr)).map(|v| v.to_u16())
    }

    fn execute_load32(&mut self, addr: u64) -> Result<u32, Error> {
        self.load32(&R::from_u64(addr)).map(|v| v.to_u32())
    }

    fn load8(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        let addr = addr.to_u64() as usize;
        check_no_overflow(addr as u64, 1, self.memory_size as u64)?;
        Ok(Self::REG::from_u8(self.data[addr]))
    }

    fn load16(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        let addr = addr.to_u64() as usize;
        check_no_overflow(addr as u64, 2, self.memory_size as u64)?;
        let v = LittleEndian::read_u16(&self.data[addr..addr + 2]);
        Ok(Self::REG::from_u16(v))
    }

    fn load32(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        let addr = addr.to_u64() as usize;
        check_no_overflow(addr as u64, 4, self.memory_size as u64)?;
        let v = LittleEndian::read_u32(&self.data[addr..addr + 4]);
        Ok(Self::REG::from_u32(v))
    }

    fn load64(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        let addr = addr.to_u64() as usize;
        check_no_overflow(addr as u64, 8, self.memory_size as u64)?;
        let v = LittleEndian::read_u64(&self.data[addr..addr + 8]);
        Ok(Self::REG::from_u64(v))
    }

    fn store8(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        let addr = addr.to_u64() as usize;
        check_no_overflow(addr as u64, 1, self.memory_size as u64)?;
        let page_indices = get_page_indices(addr as u64, 1);
        set_dirty(self, &page_indices)?;
        self.data[addr] = value.to_u8();
        Ok(())
    }

    fn store16(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        let addr = addr.to_u64() as usize;
        check_no_overflow(addr as u64, 2, self.memory_size as u64)?;
        let page_indices = get_page_indices(addr as u64, 2);
        set_dirty(self, &page_indices)?;
        LittleEndian::write_u16(&mut self.data[addr..addr + 2], value.to_u16());
        Ok(())
    }

    fn store32(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        let addr = addr.to_u64() as usize;
        check_no_overflow(addr as u64, 4, self.memory_size as u64)?;
        let page_indices = get_page_indices(addr as u64, 4);
        set_dirty(self, &page_indices)?;
        LittleEndian::write_u32(&mut self.data[addr..addr + 4], value.to_u32());
        Ok(())
    }

    fn store64(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        let addr = addr.to_u64() as usize;
        check_no_overflow(addr as u64, 8, self.memory_size as u64)?;
        let page_indices = get_page_indices(addr as u64, 8);
        set_dirty(self, &page_indices)?;
        LittleEndian::write_u64(&mut self.data[addr..addr + 8], value.to_u64());
        Ok(())
    }

    fn store_bytes(&mut self, addr: u64, value: &[u8]) -> Result<(), Error> {
        let size = value.len() as u64;
        if size == 0 {
            return Ok(());
        }
        check_no_overflow(addr, size, self.memory_size as u64)?;
        let page_indices = get_page_indices(addr, size);
        set_dirty(self, &page_indices)?;
        let slice = &mut self[addr as usize..(addr + size) as usize];
        slice.copy_from_slice(value);
        Ok(())
    }

    fn store_byte(&mut self, addr: u64, size: u64, value: u8) -> Result<(), Error> {
        if size == 0 {
            return Ok(());
        }
        check_no_overflow(addr, size, self.memory_size as u64)?;
        let page_indices = get_page_indices(addr, size);
        set_dirty(self, &page_indices)?;
        memset(&mut self[addr as usize..(addr + size) as usize], value);
        Ok(())
    }

    fn load_bytes(&mut self, addr: u64, size: u64) -> Result<Bytes, Error> {
        if size == 0 {
            return Ok(Bytes::new());
        }
        check_no_overflow(addr, size, self.memory_size as u64)?;
        Ok(Bytes::from(
            self[addr as usize..(addr + size) as usize].to_vec(),
        ))
    }

    fn lr(&self) -> &Self::REG {
        &self.load_reservation_address
    }

    fn set_lr(&mut self, value: &Self::REG) {
        self.load_reservation_address = value.clone();
    }
}
