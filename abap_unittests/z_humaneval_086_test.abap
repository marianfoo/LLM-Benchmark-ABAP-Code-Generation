CLASS ltcl_humaneval_086 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PUBLIC SECTION.
    METHODS:
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING.
ENDCLASS.

CLASS ltcl_humaneval_086 IMPLEMENTATION.

  METHOD test_case_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_086=>anti_shuffle( 'Hi' )
      exp = 'Hi' ).
  ENDMETHOD.

  METHOD test_case_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_086=>anti_shuffle( 'hello' )
      exp = 'ehllo' ).
  ENDMETHOD.

  METHOD test_case_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_086=>anti_shuffle( 'number' )
      exp = 'bemnru' ).
  ENDMETHOD.

  METHOD test_case_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_086=>anti_shuffle( 'abcd' )
      exp = 'abcd' ).
  ENDMETHOD.

  METHOD test_case_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_086=>anti_shuffle( 'Hello World!!!' )
      exp = 'Hello !!!Wdlor' ).
  ENDMETHOD.

  METHOD test_case_6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_086=>anti_shuffle( '' )
      exp = '' ).
  ENDMETHOD.

  METHOD test_case_7.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_086=>anti_shuffle( 'Hi. My name is Mister Robot. How are you?' )
      exp = '.Hi My aemn is Meirst .Rboot How aer ?ouy' ).
  ENDMETHOD.

ENDCLASS.