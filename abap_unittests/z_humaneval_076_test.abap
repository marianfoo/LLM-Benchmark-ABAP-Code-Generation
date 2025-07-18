CLASS ltcl_test_is_simple_power DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_16_2 FOR TESTING,
      test_143214_16 FOR TESTING,
      test_4_2 FOR TESTING,
      test_9_3 FOR TESTING,
      test_16_4 FOR TESTING,
      test_24_2 FOR TESTING,
      test_128_4 FOR TESTING,
      test_12_6 FOR TESTING,
      test_1_1 FOR TESTING,
      test_1_12 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_is_simple_power IMPLEMENTATION.

  METHOD test_16_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_076=>is_simple_power( x = 16 n = 2 )
      exp = abap_true
      msg = 'This prints if this assert fails 1 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_143214_16.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_076=>is_simple_power( x = 143214 n = 16 )
      exp = abap_false
      msg = 'This prints if this assert fails 1 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_4_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_076=>is_simple_power( x = 4 n = 2 )
      exp = abap_true
      msg = 'This prints if this assert fails 1 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_9_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_076=>is_simple_power( x = 9 n = 3 )
      exp = abap_true
      msg = 'This prints if this assert fails 1 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_16_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_076=>is_simple_power( x = 16 n = 4 )
      exp = abap_true
      msg = 'This prints if this assert fails 1 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_24_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_076=>is_simple_power( x = 24 n = 2 )
      exp = abap_false
      msg = 'This prints if this assert fails 1 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_128_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_076=>is_simple_power( x = 128 n = 4 )
      exp = abap_false
      msg = 'This prints if this assert fails 1 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_12_6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_076=>is_simple_power( x = 12 n = 6 )
      exp = abap_false
      msg = 'This prints if this assert fails 1 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_1_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_076=>is_simple_power( x = 1 n = 1 )
      exp = abap_true
      msg = 'This prints if this assert fails 2 (also good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_1_12.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_076=>is_simple_power( x = 1 n = 12 )
      exp = abap_true
      msg = 'This prints if this assert fails 2 (also good for debugging!)'
    ).
  ENDMETHOD.

ENDCLASS.