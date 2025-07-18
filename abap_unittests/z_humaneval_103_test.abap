CLASS ltc_test_humaneval_103 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_case_1 FOR TESTING,
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

CLASS ltc_test_humaneval_103 IMPLEMENTATION.

  METHOD test_case_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 1 m = 5 )
      exp = '0b11' ).
  ENDMETHOD.

  METHOD test_case_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 7 m = 13 )
      exp = '0b1010' ).
  ENDMETHOD.

  METHOD test_case_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 964 m = 977 )
      exp = '0b1111001010' ).
  ENDMETHOD.

  METHOD test_case_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 996 m = 997 )
      exp = '0b1111100100' ).
  ENDMETHOD.

  METHOD test_case_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 560 m = 851 )
      exp = '0b1011000010' ).
  ENDMETHOD.

  METHOD test_case_6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 185 m = 546 )
      exp = '0b101101110' ).
  ENDMETHOD.

  METHOD test_case_7.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 362 m = 496 )
      exp = '0b110101101' ).
  ENDMETHOD.

  METHOD test_case_8.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 350 m = 902 )
      exp = '0b1001110010' ).
  ENDMETHOD.

  METHOD test_case_9.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 197 m = 233 )
      exp = '0b11010111' ).
  ENDMETHOD.

  METHOD test_case_10.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 7 m = 5 )
      exp = '-1' ).
  ENDMETHOD.

  METHOD test_case_11.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 5 m = 1 )
      exp = '-1' ).
  ENDMETHOD.

  METHOD test_case_12.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_103=>rounded_avg( n = 5 m = 5 )
      exp = '0b101' ).
  ENDMETHOD.

ENDCLASS.