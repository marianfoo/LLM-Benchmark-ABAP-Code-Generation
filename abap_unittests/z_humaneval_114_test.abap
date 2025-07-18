CLASS ltcl_test_min_sub_array_sum DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING,
      test_case_11 FOR TESTING,
      test_case_12 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_min_sub_array_sum IMPLEMENTATION.

  METHOD test_case_1.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND 2 TO nums.
    APPEND 3 TO nums.
    APPEND 4 TO nums.
    APPEND 1 TO nums.
    APPEND 2 TO nums.
    APPEND 4 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND -1 TO nums.
    APPEND -2 TO nums.
    APPEND -3 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -6 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND -1 TO nums.
    APPEND -2 TO nums.
    APPEND -3 TO nums.
    APPEND 2 TO nums.
    APPEND -10 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -14 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND -9999999 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -9999999 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND 0 TO nums.
    APPEND 10 TO nums.
    APPEND 20 TO nums.
    APPEND 1000000 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND -1 TO nums.
    APPEND -2 TO nums.
    APPEND -3 TO nums.
    APPEND 10 TO nums.
    APPEND -5 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -6 ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND 100 TO nums.
    APPEND -1 TO nums.
    APPEND -2 TO nums.
    APPEND -3 TO nums.
    APPEND 10 TO nums.
    APPEND -5 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -6 ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND 10 TO nums.
    APPEND 11 TO nums.
    APPEND 13 TO nums.
    APPEND 8 TO nums.
    APPEND 3 TO nums.
    APPEND 4 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 3 ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND 100 TO nums.
    APPEND -33 TO nums.
    APPEND 32 TO nums.
    APPEND -1 TO nums.
    APPEND 0 TO nums.
    APPEND -2 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -33 ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND -10 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -10 ).
  ENDMETHOD.

  METHOD test_case_11.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND 7 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 7 ).
  ENDMETHOD.

  METHOD test_case_12.
    DATA: nums   TYPE TABLE OF i,
          result TYPE i.

    APPEND 1 TO nums.
    APPEND -1 TO nums.

    result = z_humaneval_114=>min_sub_array_sum( nums ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

ENDCLASS.