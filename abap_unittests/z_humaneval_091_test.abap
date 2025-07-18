CLASS ztc_humaneval_091 DEFINITION FINAL FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS:
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING.
ENDCLASS.

CLASS ztc_humaneval_091 IMPLEMENTATION.

  METHOD test_case_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_091=>is_bored( 'Hello world' )
      exp = 0
      msg = 'Test 1 failed' ).
  ENDMETHOD.

  METHOD test_case_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_091=>is_bored( 'Is the sky blue?' )
      exp = 0
      msg = 'Test 2 failed' ).
  ENDMETHOD.

  METHOD test_case_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_091=>is_bored( 'I love It !' )
      exp = 1
      msg = 'Test 3 failed' ).
  ENDMETHOD.

  METHOD test_case_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_091=>is_bored( 'bIt' )
      exp = 0
      msg = 'Test 4 failed' ).
  ENDMETHOD.

  METHOD test_case_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_091=>is_bored( 'I feel good today. I will be productive. will kill It' )
      exp = 2
      msg = 'Test 5 failed' ).
  ENDMETHOD.

  METHOD test_case_6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_091=>is_bored( 'You and I are going for a walk' )
      exp = 0
      msg = 'Test 6 failed' ).
  ENDMETHOD.

ENDCLASS.