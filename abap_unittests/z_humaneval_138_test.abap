CLASS ltcl_test_is_equal_to_sum_even DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_4 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_10 FOR TESTING,
      test_case_11 FOR TESTING,
      test_case_12 FOR TESTING,
      test_case_13 FOR TESTING,
      test_case_16 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_is_equal_to_sum_even IMPLEMENTATION.

  METHOD test_case_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_138=>is_equal_to_sum_even( 4 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_138=>is_equal_to_sum_even( 6 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_8.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_138=>is_equal_to_sum_even( 8 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_10.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_138=>is_equal_to_sum_even( 10 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_11.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_138=>is_equal_to_sum_even( 11 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_12.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_138=>is_equal_to_sum_even( 12 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_13.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_138=>is_equal_to_sum_even( 13 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_16.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_138=>is_equal_to_sum_even( 16 )
      exp = abap_true ).
  ENDMETHOD.

ENDCLASS.