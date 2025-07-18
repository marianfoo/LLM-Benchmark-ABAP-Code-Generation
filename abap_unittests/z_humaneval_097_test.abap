CLASS ltcl_test_multiply DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_multiply_1 FOR TESTING,
      test_multiply_2 FOR TESTING,
      test_multiply_3 FOR TESTING,
      test_multiply_4 FOR TESTING,
      test_multiply_5 FOR TESTING,
      test_multiply_6 FOR TESTING,
      test_multiply_edge_1 FOR TESTING,
      test_multiply_edge_2 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_multiply IMPLEMENTATION.

  METHOD test_multiply_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_097=>multiply( a = 148 b = 412 )
      exp = 16
      msg = 'First test error'
    ).
  ENDMETHOD.

  METHOD test_multiply_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_097=>multiply( a = 19 b = 28 )
      exp = 72
      msg = 'Second test error'
    ).
  ENDMETHOD.

  METHOD test_multiply_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_097=>multiply( a = 2020 b = 1851 )
      exp = 0
      msg = 'Third test error'
    ).
  ENDMETHOD.

  METHOD test_multiply_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_097=>multiply( a = 14 b = -15 )
      exp = 20
      msg = 'Fourth test error'
    ).
  ENDMETHOD.

  METHOD test_multiply_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_097=>multiply( a = 76 b = 67 )
      exp = 42
      msg = 'Fifth test error'
    ).
  ENDMETHOD.

  METHOD test_multiply_6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_097=>multiply( a = 17 b = 27 )
      exp = 49
      msg = 'Sixth test error'
    ).
  ENDMETHOD.

  METHOD test_multiply_edge_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_097=>multiply( a = 0 b = 1 )
      exp = 0
      msg = '1st edge test error'
    ).
  ENDMETHOD.

  METHOD test_multiply_edge_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_097=>multiply( a = 0 b = 0 )
      exp = 0
      msg = '2nd edge test error'
    ).
  ENDMETHOD.

ENDCLASS.