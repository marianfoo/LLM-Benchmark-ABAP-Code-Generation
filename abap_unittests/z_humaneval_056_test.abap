CLASS ltcl_test_correct_bracketing DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_simple_pair FOR TESTING,
      test_nested_brackets FOR TESTING,
      test_multiple_pairs FOR TESTING,
      test_complex_nested FOR TESTING,
      test_unbalanced_nested FOR TESTING,
      test_wrong_order FOR TESTING,
      test_single_opening FOR TESTING,
      test_multiple_opening FOR TESTING,
      test_single_closing FOR TESTING,
      test_incomplete_pair FOR TESTING,
      test_extra_closing_1 FOR TESTING,
      test_extra_closing_2 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_correct_bracketing IMPLEMENTATION.
  METHOD test_simple_pair.
    cl_abap_unit_assert=>assert_true( z_humaneval_056=>correct_bracketing( '<>' ) ).
  ENDMETHOD.

  METHOD test_nested_brackets.
    cl_abap_unit_assert=>assert_true( z_humaneval_056=>correct_bracketing( '<<><>>' ) ).
  ENDMETHOD.

  METHOD test_multiple_pairs.
    cl_abap_unit_assert=>assert_true( z_humaneval_056=>correct_bracketing( '<><><<><>><>' ) ).
  ENDMETHOD.

  METHOD test_complex_nested.
    cl_abap_unit_assert=>assert_true( z_humaneval_056=>correct_bracketing( '<><><<<><><>><>><<><><<>>>' ) ).
  ENDMETHOD.

  METHOD test_unbalanced_nested.
    cl_abap_unit_assert=>assert_false( z_humaneval_056=>correct_bracketing( '<<<><>>>>' ) ).
  ENDMETHOD.

  METHOD test_wrong_order.
    cl_abap_unit_assert=>assert_false( z_humaneval_056=>correct_bracketing( '><<>' ) ).
  ENDMETHOD.

  METHOD test_single_opening.
    cl_abap_unit_assert=>assert_false( z_humaneval_056=>correct_bracketing( '<' ) ).
  ENDMETHOD.

  METHOD test_multiple_opening.
    cl_abap_unit_assert=>assert_false( z_humaneval_056=>correct_bracketing( '<<<<' ) ).
  ENDMETHOD.

  METHOD test_single_closing.
    cl_abap_unit_assert=>assert_false( z_humaneval_056=>correct_bracketing( '>' ) ).
  ENDMETHOD.

  METHOD test_incomplete_pair.
    cl_abap_unit_assert=>assert_false( z_humaneval_056=>correct_bracketing( '<<>' ) ).
  ENDMETHOD.

  METHOD test_extra_closing_1.
    cl_abap_unit_assert=>assert_false( z_humaneval_056=>correct_bracketing( '<><><<><>><>><<>' ) ).
  ENDMETHOD.

  METHOD test_extra_closing_2.
    cl_abap_unit_assert=>assert_false( z_humaneval_056=>correct_bracketing( '<><><<><>><>>><>' ) ).
  ENDMETHOD.
ENDCLASS.