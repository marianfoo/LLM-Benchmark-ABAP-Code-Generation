CLASS ltcl_test_words_string DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_basic_case_1 FOR TESTING,
      test_basic_case_2 FOR TESTING,
      test_basic_case_3 FOR TESTING,
      test_basic_case_4 FOR TESTING,
      test_edge_case_1 FOR TESTING,
      test_edge_case_2 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_words_string IMPLEMENTATION.

  METHOD test_basic_case_1.
    DATA: lv_result   TYPE string_table,
          lv_expected TYPE string_table.

    lv_result = z_humaneval_101=>words_string( 'Hi, my name is John' ).
    APPEND 'Hi' TO lv_expected.
    APPEND 'my' TO lv_expected.
    APPEND 'name' TO lv_expected.
    APPEND 'is' TO lv_expected.
    APPEND 'John' TO lv_expected.

    cl_abap_unit_assert=>assert_equals( act = lv_result exp = lv_expected ).
  ENDMETHOD.

  METHOD test_basic_case_2.
    DATA: lv_result   TYPE string_table,
          lv_expected TYPE string_table.

    CLEAR: lv_result, lv_expected.
    lv_result = z_humaneval_101=>words_string( 'One, two, three, four, five, six' ).
    APPEND 'One' TO lv_expected.
    APPEND 'two' TO lv_expected.
    APPEND 'three' TO lv_expected.
    APPEND 'four' TO lv_expected.
    APPEND 'five' TO lv_expected.
    APPEND 'six' TO lv_expected.

    cl_abap_unit_assert=>assert_equals( act = lv_result exp = lv_expected ).
  ENDMETHOD.

  METHOD test_basic_case_3.
    DATA: lv_result   TYPE string_table,
          lv_expected TYPE string_table.

    CLEAR: lv_result, lv_expected.
    lv_result = z_humaneval_101=>words_string( 'Hi, my name' ).
    APPEND 'Hi' TO lv_expected.
    APPEND 'my' TO lv_expected.
    APPEND 'name' TO lv_expected.

    cl_abap_unit_assert=>assert_equals( act = lv_result exp = lv_expected ).
  ENDMETHOD.

  METHOD test_basic_case_4.
    DATA: lv_result   TYPE string_table,
          lv_expected TYPE string_table.

    CLEAR: lv_result, lv_expected.
    lv_result = z_humaneval_101=>words_string( 'One,, two, three, four, five, six,' ).
    APPEND 'One' TO lv_expected.
    APPEND 'two' TO lv_expected.
    APPEND 'three' TO lv_expected.
    APPEND 'four' TO lv_expected.
    APPEND 'five' TO lv_expected.
    APPEND 'six' TO lv_expected.

    cl_abap_unit_assert=>assert_equals( act = lv_result exp = lv_expected ).
  ENDMETHOD.

  METHOD test_edge_case_1.
    DATA: lv_result   TYPE string_table,
          lv_expected TYPE string_table.

    lv_result = z_humaneval_101=>words_string( '' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = lv_expected ).
  ENDMETHOD.

  METHOD test_edge_case_2.
    DATA: lv_result   TYPE string_table,
          lv_expected TYPE string_table.

    CLEAR: lv_result, lv_expected.
    lv_result = z_humaneval_101=>words_string( 'ahmed     , gamal' ).
    APPEND 'ahmed' TO lv_expected.
    APPEND 'gamal' TO lv_expected.

    cl_abap_unit_assert=>assert_equals( act = lv_result exp = lv_expected ).
  ENDMETHOD.

ENDCLASS.