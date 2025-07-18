CLASS ltcl_test_correct_bracketing DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

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

CLASS ltcl_test_correct_bracketing IMPLEMENTATION.

  METHOD test_case_1.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( '()' ).
    cl_abap_unit_assert=>assert_true( result ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( '(()())' ).
    cl_abap_unit_assert=>assert_true( result ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( '()()(()())()' ).
    cl_abap_unit_assert=>assert_true( result ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( '()()((()()())())(()()(()))' ).
    cl_abap_unit_assert=>assert_true( result ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( '((()()))' ).
    cl_abap_unit_assert=>assert_true( result ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( ')(()' ).
    cl_abap_unit_assert=>assert_false( result ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( '(' ).
    cl_abap_unit_assert=>assert_false( result ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( '((((' ).
    cl_abap_unit_assert=>assert_false( result ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( ')' ).
    cl_abap_unit_assert=>assert_false( result ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( '(()' ).
    cl_abap_unit_assert=>assert_false( result ).
  ENDMETHOD.

  METHOD test_case_11.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( '()()(()())())((' ).
    cl_abap_unit_assert=>assert_false( result ).
  ENDMETHOD.

  METHOD test_case_12.
    DATA: result TYPE abap_bool.
    result = z_humaneval_061=>correct_bracketing( '()()(()())()))()' ).
    cl_abap_unit_assert=>assert_false( result ).
  ENDMETHOD.

ENDCLASS.