CLASS ltc_test_count_upper DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_count_upper_case_1 FOR TESTING,
      test_count_upper_case_2 FOR TESTING,
      test_count_upper_case_3 FOR TESTING,
      test_count_upper_case_4 FOR TESTING,
      test_count_upper_case_5 FOR TESTING,
      test_count_upper_case_6 FOR TESTING,
      test_count_upper_case_7 FOR TESTING.

ENDCLASS.

CLASS ltc_test_count_upper IMPLEMENTATION.

  METHOD test_count_upper_case_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_098=>count_upper( 'aBCdEf' )
      exp = 1 ).
  ENDMETHOD.

  METHOD test_count_upper_case_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_098=>count_upper( 'abcdefg' )
      exp = 0 ).
  ENDMETHOD.

  METHOD test_count_upper_case_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_098=>count_upper( 'dBBE' )
      exp = 0 ).
  ENDMETHOD.

  METHOD test_count_upper_case_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_098=>count_upper( 'B' )
      exp = 0 ).
  ENDMETHOD.

  METHOD test_count_upper_case_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_098=>count_upper( 'U' )
      exp = 1 ).
  ENDMETHOD.

  METHOD test_count_upper_case_6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_098=>count_upper( '' )
      exp = 0 ).
  ENDMETHOD.

  METHOD test_count_upper_case_7.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_098=>count_upper( 'EEEE' )
      exp = 2 ).
  ENDMETHOD.

ENDCLASS.