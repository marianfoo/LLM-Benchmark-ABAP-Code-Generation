CLASS ltcl_test_double_difference DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: tt_int_table TYPE TABLE OF i.
    METHODS: test_empty_list FOR TESTING,
      test_simple_case FOR TESTING,
      test_negative_numbers FOR TESTING,
      test_mixed_case FOR TESTING,
      test_odd_numbers FOR TESTING,
      test_range_case FOR TESTING.
ENDCLASS.

CLASS ltcl_test_double_difference IMPLEMENTATION.
  METHOD test_empty_list.
    DATA: lst    TYPE tt_int_table,
          result TYPE i.

    result = z_humaneval_151=>double_the_difference( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_simple_case.
    DATA: lst    TYPE tt_int_table,
          result TYPE i.

    APPEND 5 TO lst.
    APPEND 4 TO lst.
    result = z_humaneval_151=>double_the_difference( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 25 ).
  ENDMETHOD.

  METHOD test_negative_numbers.
    DATA: lst    TYPE tt_int_table,
          result TYPE i.

    APPEND -10 TO lst.
    APPEND -20 TO lst.
    APPEND -30 TO lst.
    result = z_humaneval_151=>double_the_difference( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_mixed_case.
    DATA: lst    TYPE tt_int_table,
          result TYPE i.

    APPEND -1 TO lst.
    APPEND -2 TO lst.
    APPEND 8 TO lst.
    result = z_humaneval_151=>double_the_difference( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_odd_numbers.
    DATA: lst    TYPE tt_int_table,
          result TYPE i.

    APPEND 3 TO lst.
    APPEND 5 TO lst.
    result = z_humaneval_151=>double_the_difference( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 34 ).
  ENDMETHOD.

  METHOD test_range_case.
    DATA: lst      TYPE tt_int_table,
          result   TYPE i,
          expected TYPE i,
          lv_num   TYPE i.

    expected = 0.
    DO 100 TIMES.
      lv_num = ( sy-index - 1 ) * 2 - 99.
      IF lv_num > 0 AND lv_num MOD 2 = 1.
        expected = expected + ( lv_num * lv_num ).
      ENDIF.
      APPEND lv_num TO lst.
    ENDDO.

    result = z_humaneval_151=>double_the_difference( lst ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.
ENDCLASS.